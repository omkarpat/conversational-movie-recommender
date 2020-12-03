import torch
from collections import defaultdict
from itertools import chain
import os


class TransferTransfoConstants:
    SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
    MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
    PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

    ATTR_TO_SPECIAL_TOKEN = {
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'pad_token': '<pad>',
        'additional_special_tokens': ["<speaker1>", "<speaker2>", "<end>", "<eot>"]
    }

    ADDITIONAL_TOKENS = ["_nofact"]

class TransferTransfoWithKnowledgeConstants(object):
    SPECIAL_TOKENS = TransferTransfoConstants.SPECIAL_TOKENS
    MODEL_INPUTS = TransferTransfoConstants.MODEL_INPUTS
    PADDED_INPUTS = TransferTransfoConstants.PADDED_INPUTS

    ADDITIONAL_TOKENS = TransferTransfoConstants.ADDITIONAL_TOKENS + [
        "<person>", "</person>",
        "<genre>", "</genre>",
        "<movie_title>", "</movie_title>"
    ]

    ATTR_TO_SPECIAL_TOKEN = TransferTransfoConstants.ATTR_TO_SPECIAL_TOKEN


def collate_batch_elements(batch, tokenizer, device, pad_left=False):

    batch_inputs = defaultdict(list)

    for instance in batch:
        for field, data in instance.items():
            batch_inputs[field].append(data)
    
    pad_token_map = {
        "labels": -100,
        "input_ids": tokenizer.eos_token_id,
        "default": 0
    }
    padded_inputs = ["input_ids", "labels", "token_type_ids"]
    
    model_inputs = ["input_ids", "labels", "token_type_ids"]

    if pad_left:
        model_inputs.append("attention_mask")
    padded_batch = pad_batch_items(batch_inputs, pad_token_map, padded_inputs, pad_left)
    tensorized_input = []

    for input_name in model_inputs:
        tensor = torch.tensor(padded_batch[input_name], device=device)
        tensorized_input.append(tensor)
    
    return tensorized_input


def collate_transfertransfo_batch_elements(batch, tokenizer, args):
    batch_inputs = defaultdict(list)
    chained_batch = chain(*batch)

    for instance in chained_batch:
        for field, data in instance.items():
            batch_inputs[field].append(data)


    pad_token = tokenizer.convert_tokens_to_ids(TransferTransfoConstants.SPECIAL_TOKENS[-2])

    padded_dataset = pad_dataset(batch_inputs, padding=pad_token)

    tensorized_input = []

    batch_size = tuple([len(batch_inputs[TransferTransfoConstants.MODEL_INPUTS[0]]) // args.num_candidates])
    for input_name in TransferTransfoConstants.MODEL_INPUTS:
        tensor = torch.tensor(padded_dataset[input_name])

        if input_name != "mc_labels":
            tensor = tensor.view((-1, args.num_candidates) + tensor.shape[1:])
        else:
            tensor = torch.ones(size=batch_size, dtype=torch.long) * (args.num_candidates - 1)
        tensorized_input.append(tensor)
    return tensorized_input


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in TransferTransfoConstants.PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def pad_batch_items(batch_items, pad_token_map, padded_inputs, pad_left):
    max_seq_len = max(len(x) for x in batch_items["input_ids"])

    default_pad_token = pad_token_map["default"]

    if pad_left:
        # Attention mask is necessary to avoid attending on left padding tokens
        # this isn't a problem for the right-padding case since
        # the logits from the right padding tokens can be ignored.
        # See: https://github.com/huggingface/transformers/issues/808
        batch_items["attention_mask"] = [[0 if i < max_seq_len - len(x) else 1 for i in range(max_seq_len)] for x in batch_items["input_ids"]]

    for name in padded_inputs:
        pad_token = pad_token_map.get(name, default_pad_token)

        if pad_left:
            # Experimenting with left padding for batch inference
            batch_items[name] = [ ([pad_token] * (max_seq_len - len(x)) + x) for x in batch_items[name]]
        else:
            batch_items[name] = [ (x + [pad_token] * (max_seq_len - len(x))) for x in batch_items[name]]
    return batch_items        


def save_model_config_and_tokenizer(config, tokenizer, args):
    config_path = os.path.join(args.experiment_path, args.experiment_name)
    os.makedirs(config_path, exist_ok=True)
    config.save_pretrained(config_path)
    tokenizer.save_pretrained(config_path)


def save_model_checkpoint(model, args, checkpoint_name="checkpoint.pt"):
    checkpoint_path = os.path.join(args.experiment_path, args.experiment_name, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file_path = os.path.join(checkpoint_path, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_file_path)


def save_full_model(model, tokenizer, args, model_name):
    checkpoint_path = os.path.join(args.experiment_path, args.experiment_name, model_name)
    
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

