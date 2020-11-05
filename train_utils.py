import torch
from collections import defaultdict
from itertools import chain
import os

def collate_batch_elements(batch, tokenizer, device):

    batch_inputs = defaultdict(list)

    for instance in batch:
        for field, data in instance.items():
            batch_inputs[field].append(data)
    
    pad_token_map = {
        "labels": -100,
        "default": 0
    }
    padded_inputs = ["input_ids", "labels", "token_type_ids"]
    model_inputs = ["input_ids", "labels", "token_type_ids"]

    padded_batch = pad_batch_items(batch_inputs, pad_token_map, padded_inputs)


    tensorized_input = []

    for input_name in model_inputs:
        tensor = torch.tensor(padded_batch[input_name], device=device)
        tensorized_input.append(tensor)
    
    return tensorized_input



def pad_batch_items(batch_items, pad_token_map, padded_inputs):
    max_seq_len = max(len(x) for x in batch_items["input_ids"])

    default_pad_token = pad_token_map["default"]

    for name in padded_inputs:
        pad_token = pad_token_map.get(name, default_pad_token)

        batch_items[name] = [ (x + [pad_token] * (max_seq_len - len(x))) for x in batch_items[name]]

    return batch_items        

def save_model_config(model, tokenizer, args):
    pass

def save_model_checkpoint(model, args, checkpoint_name="checkpoint.pt"):
    checkpoint_path = os.path.join(args.experiment_path, args.experiment_name, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_path = os.path.join(args.experiment_path, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)

def save_full_model(model, args, model_name):
    checkpoint_path = os.path.join(args.experiment_path, model_name)
    
    model.save_pretrained(save_model_checkpoint)
