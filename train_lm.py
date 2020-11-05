import argparse
from tqdm.auto import tqdm

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import RedialDialoGPTDataset
from dataset_utils import (
    prepare_redial_baseline_dataset, 
    get_movie_db_map
    )

from trainer.metrics import RunningMetric

from train_utils import (
    collate_batch_elements,
    save_model_config,
    save_model_checkpoint
)

def prepare_dataloaders(args, tokenizer):

    movie_db_map = get_movie_db_map(args.movies_data_path)
    dataset = prepare_redial_baseline_dataset(
        args.data_path,
        tokenizer,
        movie_db_map,
        args.data_cache_path
        )

    train_dataset, test_dataset = RedialDialoGPTDataset(dataset["train"], tokenizer, args), \
        RedialDialoGPTDataset(dataset["test"], tokenizer, args)


    collate_fn = lambda batch: collate_batch_elements(batch, tokenizer, args.device)

    train_loader, test_loader = \
        DataLoader(
            train_dataset, 
            batch_size=args.train_batch_size, 
            collate_fn=collate_fn,
            shuffle=True), \
        DataLoader(
            test_dataset, 
            batch_size=args.test_batch_size,
            collate_fn=collate_fn,
            shuffle=False)

    return train_loader, test_loader

def train_lm(model, loader, optimizer, args):
    model.train()

    running_loss = RunningMetric()

    for i, batch in tqdm(enumerate(loader)):
        input_ids, labels, token_type_ids = batch

        # Forward pass and loss computation
        lm_loss, *_ = model(input_ids=input_ids, labels=labels, token_type_ids=token_type_ids)
        # lm_loss is given by sum(batch_loss) / batch_size

        # We need to average loss over all gradient accumulation steps
        loss = lm_loss / args.gradient_accumulation_steps

        running_loss.add(loss.item())

        # Backprop step. Pytorch automatically accumulates
        # 
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        # Parameter update step
        if (i + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    print(f"Epoch loss: {running_loss.get()}")
    print(f"Epoch PPL: {math.exp(running_loss.get())}")

def evaluate_lm(model, loader, loss_fn, args):
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            input_ids, labels, token_type_ids = batch

            # No loss is returned if lm_labels is not passed
            lm_logits, *_ = model(
                input_ids=input_ids, token_type_ids=token_type_ids
            )

            lm_logits_flattened = lm_logits.view(-1, lm_logits.size(-1))
            labels_flattened = labels.view(-1)
            loss = loss_fn(lm_logits_flattened, labels_flattened)



def train_baseline_lm(model, loaders, optimizer, loss_fn, args):
    
    train_loader, test_loader = loaders

    for i in range(args.n_epochs):
        train_lm(model, train_loader, optimizer, args)        
        evaluate_lm(model, test_loader, optimizer, args)

    if args.n_epochs < 1:
        evaluate_lm(model, test_loader, optimizer, args)

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint',
        default="gpt2-medium",
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
    parser.add_argument('--seed',
        default=42,
        type=int,
        help="Random seed for experiments"
    )

    parser.add_argument('--max_history_turns',
        default=2,
        type=int,
        help="How many turns from context to retain"
    )

    parser.add_argument('--n_epochs',
        default=3,
        type=int,
        help="Number of epochs for training"
    )

    parser.add_argument('--train_batch_size',
        default=4,
        type=int,
        help="Train batch size"
    )
    parser.add_argument('--test_batch_size',
        default=4,
        type=int,
        help="Test batch size"
    )
    parser.add_argument('--lr',
        default=6.25e-5,
        type=float,
        help="Base/initial learning rate"
    )
    parser.add_argument('--device',
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to store the model"
    )
    parser.add_argument('--gradient_accumulation_steps',
        default=16,
        type=int,
        help="Number of steps to accumulate gradient for"
    )
    parser.add_argument('--max_norm',
        default=1.0,
        type=float,
        help="Maximum norm for the gradient"
    )

    return parser

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    train_loader, test_loader = prepare_dataloaders(args, tokenizer)
    
    model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    train_baseline_lm(model, (train_loader, test_loader), optimizer, loss_fn, args)