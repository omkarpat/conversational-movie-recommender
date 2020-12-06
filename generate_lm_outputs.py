import argparse
import torch

from tqdm.auto import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import RedialDialoGPTDataset, RedialTransferTransfoDataset
from dataset_utils import get_movie_db_map, prepare_redial_baseline_dataset, prepare_redial_knowledge_grounded_dataset
from train_utils import collate_batch_elements, collate_transfertransfo_batch_elements, TransferTransfoConstants

def prepare_dataloader(args, tokenizer):

    movie_db_map = get_movie_db_map(args.movies_data_path)
    dataset = prepare_redial_baseline_dataset(
        args.data_path,
        tokenizer,
        movie_db_map,
        args.data_cache_path
        )

    test_dataset = RedialDialoGPTDataset(dataset["test"], tokenizer, args)


    collate_fn = lambda batch: collate_batch_elements(batch, tokenizer, args.device, pad_left=True)

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn, shuffle=False)

    return test_loader

def prepare_knowledge_grounded_dataloader(args, tokenizer):
    movie_db_map = get_movie_db_map(args.movies_data_path)
    dataset, special_terms = prepare_redial_knowledge_grounded_dataset(
        args.data_path,
        tokenizer,
        movie_db_map,
        args.data_cache_path,
        split_files={"train": args.train_file, "test": args.eval_file},
        recommender_only=args.recommender_only,
        include_dacts=args.include_dialog_acts
    )
    special_terms.extend([
        "<cast>", "</cast>",
        "<movie_genre>", "</movie_genre>",
        "<director>", "</director>",
    ])

    test_dataset = RedialTransferTransfoDataset(
        dataset["test"], tokenizer, TransferTransfoConstants.SPECIAL_TOKENS, args)

    def collate_fn(batch):
        return collate_transfertransfo_batch_elements(batch, tokenizer, args)

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn, shuffle=False)

    return test_loader


def save_outputs(outputs, args):
    with open(args.output_file_path, 'w') as output_file:
        output_file.write("\n".join(outputs))


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code

    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def decode_sequences(input_ids, token_type_ids, model, tokenizer, args):

    special_tokens_ids = tokenizer.convert_tokens_to_ids(TransferTransfoConstants.SPECIAL_TOKENS)

    outputs = []
    for i in range(len(input_ids)):
        input_seq = tokenizer.decode(input_ids[i][0])
        prefix, suffix = input_seq.rsplit("<speaker", maxsplit=1)
        context = prefix + "<speaker" + suffix[:2]  # Hacky way to append the speaker tag
        current_output = []

        attempts = 0
        # Keep trying to generate output until a limited number of times
        expanded_tok_type_ids = token_type_ids[i][0].tolist()
        for j in range(args.max_length):  # Add trailing tokens
            expanded_tok_type_ids.append(expanded_tok_type_ids[-1])
        expanded_tok_type_ids = torch.tensor(expanded_tok_type_ids).to(args.device)
        for j in range(args.max_length):
            prefix_input_seq = torch.tensor(tokenizer.encode(context) + current_output).unsqueeze(0)
            truncated_tok_type_ids = expanded_tok_type_ids[:prefix_input_seq.shape[-1]].unsqueeze(0)
            logits = model(prefix_input_seq.to(args.device), token_type_ids=truncated_tok_type_ids.to(args.device))

            if isinstance(logits, tuple) or len(logits.shape) == 4:  # for gpt2 and maybe others
                logits = logits[0]
            logits = logits[0, -1, :] / args.temperature
            logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
            if prev.item() in special_tokens_ids:
                patience = 3
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1 or patience == 0:
                        # Disabled this rather noisy warning
                        # logger.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)
                    patience -= 1
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())

        output = tokenizer.decode(current_output)
        outputs.append(output.replace('\n', ''))

    return outputs


def generate_outputs(model, loader, tokenizer, args):

    
    all_outputs = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):

            input_ids, _, _, attention_mask = batch
            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=not args.no_sample,
                pad_token_id=tokenizer.eos_token_id,
                max_length=input_ids.size(1) + args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
                )

            for x in output_sequences:
                all_outputs.append(tokenizer.decode(x[input_ids.size(1):], skip_special_tokens=True))
    
    save_outputs(all_outputs, args)

def generate_outputs_single_example(model, loader, tokenizer, args):
    outputs = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            outputs += decode_sequences(input_ids, token_type_ids, model, tokenizer, args)

    save_outputs(outputs, args)

def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_configuration)

    if args.configuration == "baseline":
        test_loader = prepare_dataloader(args, tokenizer)
    else:
        test_loader = prepare_knowledge_grounded_dataloader(args, tokenizer)
    
    if args.model_configuration != args.model_checkpoint:
        config = GPT2Config.from_pretrained(args.model_configuration)
        state_dict = torch.load(args.model_checkpoint)
        # I didnt' read the documentation carefully enough
        model = GPT2LMHeadModel.from_pretrained(None, config=config,state_dict=state_dict)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    model.to(args.device)

    if args.configuration == "baseline":
        generate_outputs(model, test_loader, tokenizer, args)
    else:
        generate_outputs_single_example(model, test_loader, tokenizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--configuration',
                        type=str,
                        default='baseline',
                        choices=['baseline', 'knowledge_grounded'])

    parser.add_argument('--model_configuration',
        default="microsoft/DialoGPT-medium",
        help="The model configuration to use"
    )

    parser.add_argument('--model_checkpoint',
        default="microsoft/DialoGPT-medium",
        help="The model checkpoint to use"
    )

    parser.add_argument('--data_path',
        default="redial/",
        help="Path to dataset"
    )

    parser.add_argument('--movies_data_path',
        default="redial/movies_with_mentions.csv",
        help="Path to movie mentions file"
    )

    parser.add_argument('--train-file',
                        default="train_data_swda_tagged.jsonl",
                        help="Name of train jsonl file."
                        )
    parser.add_argument('--eval-file',
                        default="test_data_swda_tagged.jsonl",
                        help="Name of test jsonl file."
                        )

    parser.add_argument('--data_cache_path',
        default="redial_dataset_cache.pkl",
        help="Path to cached data"
    )

    parser.add_argument('--test_batch_size',
        default=1,
        type=int,
        help="Test batch size"
    )

    parser.add_argument('--model_metadata_path', type=str, default='./runs/bert_swbd_pd_nrg',
                        help='Path to the tokenizer and model configuration')
    parser.add_argument('--output_file_path', type=str, default='submissions/submissions.txt')

    parser.add_argument('--max_history_turns',
        default=2,
        type=int,
        help="How many turns from context to retain"
    )

    parser.add_argument('--device',
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to store the model"
    )

    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int,
                        default=0.,
                        help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")

    double_heads_parser = parser.add_argument_group('Double Heads Model Arguments:')
    double_heads_parser.add_argument('--num_candidates',
                                     type=int, default=1,
                                     help="Number of candidates to select from")
    parser.add_argument('--recommender_only',
                        dest='recommender_only',
                        action='store_true',
                        help="Train only on recommender side utterances"
                        )
    parser.set_defaults(include_dialog_acts=True)
    parser.add_argument('--exclude_dialog_acts',
                        dest='include_dialog_acts',
                        action='store_false',
                        help="Whether to exclude dialog act in the knowledge")


    args = parser.parse_args()

    args.inference = True
    main(args)