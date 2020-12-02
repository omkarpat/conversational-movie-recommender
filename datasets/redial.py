import random

from torch.utils.data import Dataset

from itertools import chain

class RedialDialoGPTDataset(Dataset):
    """
    This dataset class prepares input in the vanilla DialoGPT format:
    
    <|endoftext|> TURN_1 <|endoftext|> TURN_2 <|endoftext|> ... <|endoftext|> TURN_N <|endoftext|>

    This baseline is expected to perform worse since there's 
    nothing for the model to condition on
    """

    def __init__(self, dataset, tokenizer, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_history_turns = args.max_history_turns
        self.inference = args.inference

    def __getitem__(self, index):

        example = self.dataset[index]

        example = self._truncate_example(example)
        instance = self.prepare_input_for_dialogpt(example, self.tokenizer, self.inference)

        return instance

    def _truncate_example(self, example):

        if len(example.context) > self.max_history_turns:
            truncated_context = example.context[-self.max_history_turns:]
            example = example._replace(context=truncated_context)

        return example

    @staticmethod
    def prepare_input_for_dialogpt(example, tokenizer, inference=False):
        
        bos, eos = tokenizer.convert_tokens_to_ids([tokenizer.bos_token, tokenizer.eos_token])
        context_turns = example.context
        system_turn = example.response
        
        input_ids = [bos]
        labels = [-100]
        
        for turn in context_turns:
            input_ids += turn + [eos]
            labels += len(turn) * [eos]
        
        if not inference: # Don't add system response for inference
            input_ids += system_turn + [eos]
            labels += system_turn + [eos]
        
        token_type_ids = [0 for _ in labels]
        instance = {
            "input_ids": input_ids,
            "labels": labels,
            "token_type_ids": token_type_ids
        }

        return instance


    def __len__(self):
        return len(self.dataset)



class RedialTransferTransfoDataset(Dataset):
    """
    Mimics the setup from the TransferTransfo paper. This setup has been
    fairly reliable for knowledge grounded models such as Topical Chats.
    """

    def __init__(self, dataset, tokenizer, special_tokens, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

        self.max_history = args.max_history_turns
        self.num_candidates = args.num_candidates

    @staticmethod
    def sample_candidates(dataset, index, num_samples):
        # Sample candidates IID
        candidates = [response for (_, response, _) in random.sample(dataset, num_samples)]

        return candidates

    @staticmethod
    def build_input_from_segments(context, response, knowledge, tokenizer, special_tokens, lm_labels):

        bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(special_tokens[:4])

        sequence = [[bos] + knowledge] + context + [response + [eos]]
        sequence = [sequence[0]] + [
            [(speaker2 if (len(sequence) - i) % 2 else speaker1)] + s for i, s in enumerate(sequence[1:])]

        instance = {}
        instance["input_ids"] = list(chain.from_iterable(sequence))
        instance["token_type_ids"] = [speaker2 if (len(sequence) - i - 1) % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        if lm_labels:
            instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        else:
            instance["lm_labels"] = [-100] * len(instance["input_ids"])

        return instance

    @staticmethod
    def truncate_inputs(context, max_context_exchanges):
        # Limit the number of context exchanges to retain
        if len(context) > (2 * max_context_exchanges + 1):
            context = context[-(2 * max_context_exchanges + 1):]

        return context

    def __getitem__(self, index):

        context, response, knowledge = self.dataset[index]

        context = self.truncate_inputs(context, self.max_history)
        tokenized_context = [self.tokenizer.encode(turn) for turn in context]
        tokenized_response = self.tokenizer.encode(response)

        candidates = self.sample_candidates(self.dataset, index, self.num_candidates - 1)

        tokenized_candidates = [self.tokenizer.encode(candidate) for candidate in candidates]
        tokenized_candidates.append(tokenized_response)

        instances = []

        for j, candidate in enumerate(tokenized_candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            instance = self.build_input_from_segments(tokenized_context, candidate, [], self.tokenizer, self.special_tokens, lm_labels)
            instances.append(instance)

        return instances

    def __len__(self):
        return len(self.dataset)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size # has the effect of performing "drop last"