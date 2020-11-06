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
