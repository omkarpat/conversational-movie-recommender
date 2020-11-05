from torch.utils.data import Dataset

from itertools import chain

class RedialDialoGPTDataset(Dataset):
    def __init__(self, dataset, tokenizer, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_history_turns = args.max_history_turns

    def __getitem__(self, index):

        example = self.dataset[index]

        example = self._truncate_example(example)
        instance = self.prepare_input_for_dialogpt(example, self.tokenizer)

        return instance

    def _truncate_example(self, example):

        if len(example.context) > self.max_history_turns:
            truncated_context = example.context[-self.max_history_turns:]
            example = example._replace(context=truncated_context)

        return example

    @staticmethod
    def prepare_input_for_dialogpt(example, tokenizer):
        
        bos, eos = tokenizer.convert_tokens_to_ids([tokenizer.bos_token, tokenizer.eos_token])
        context_turns = example.context
        system_turn = example.response
        
        input_ids = [bos]
        labels = [-100]
        
        for turn in context_turns:
            input_ids += turn + [eos]
            labels += len(turn) * [eos]
        
        input_ids += system_turn + [eos]
        labels += system_turn + [-100]
        token_type_ids = [0 for _ in labels]
        instance = {
            "input_ids": input_ids,
            "labels": labels,
            "token_type_ids": token_type_ids
        }

        return instance


    def __len__(self):
        return len(self.dataset)
