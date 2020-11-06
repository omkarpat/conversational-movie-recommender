import argparse
import torch

from tqdm.auto import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from torch.utils.data import DataLoader
from datasets import RedialDialoGPTDataset
from dataset_utils import get_movie_db_map, prepare_redial_baseline_dataset
from train_utils import collate_batch_elements

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

def save_outputs(outputs, args):
    with open(args.output_file_path, 'w') as output_file:
        output_file.write("\n".join(outputs))

def generate_outputs(model, loader, tokenizer, args):

    
    all_outputs = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):

            input_ids, _, _, attention_mask = batch
            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                max_length=input_ids.size(1) + args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
                )

            for x in output_sequences:
                all_outputs.append(tokenizer.decode(x[input_ids.size(1):], skip_special_tokens=True))
    
    save_outputs(all_outputs, args)


def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    test_loader = prepare_dataloader(args, tokenizer)

    
    if args.model_configuration != args.model_checkpoint:
        config = GPT2Config.from_pretrained(args.model_configuration)
        state_dict = torch.load(args.model_checkpoint)

        model = GPT2LMHeadModel.from_pretrained(config=config,state_dict=state_dict)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    generate_outputs(model, test_loader, tokenizer, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    parser.add_argument('--data_cache_path',
        default="redial_dataset_cache.pkl",
        help="Path to cached data"
    )

    parser.add_argument('--test_batch_size',
        default=4,
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
    parser.add_argument("--top_k", type=int, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")

    args = parser.parse_args()

    args.inference = True
    main(args)