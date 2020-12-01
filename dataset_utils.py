import csv
import json
import os
import pickle

import re


from collections import namedtuple
from transformers import GPT2Tokenizer

from tqdm.auto import tqdm

BaselineExample = namedtuple(
    'BaselineExample',
    ['context', 'response']
)



def prepare_baseline_redial_split(
    split_path,
    tokenizer,
    movie_db_map
):

    with open(split_path, 'r') as split_file:
        split_conversations = split_file.read().splitlines()


    examples = []

    # Matching for movie mention ids: @1234
    movie_mention_pattern = re.compile(r"@(\d+)")
    # Pattern for mathching the year portion: (2007)
    movie_title_year_pattern = re.compile(r"\s+\(\d+\)")
    for conversation_str in tqdm(split_conversations):
        conversation = json.loads(conversation_str)

        context = []

        messages = conversation["messages"]

        response = ""

        for i, message in enumerate(messages):
            processed_text = message["text"]
            
            for mention in movie_mention_pattern.finditer(processed_text):
                movie_id = mention.group(1)
                
                # Remove year from title
                movie_title = movie_title_year_pattern.sub('', movie_db_map[movie_id])
                # for now, naively substitute movie title in message
                processed_text = processed_text.replace("@" + movie_id, movie_title)

            if i == len(messages) - 1 or \
                message["senderWorkerId"] != messages[i + 1]["senderWorkerId"]:
                response += processed_text
                encoded_response = tokenizer.encode(response)
                examples.append(BaselineExample(
                    context,
                    encoded_response
                ))

                context = context + [encoded_response]
                response = ""
            else:
                # We looked ahead and saw another follow-on response
                response += processed_text + " . "

    return examples


def get_movie_db_map(mentions_file_path):
    movie_db_map = {}

    with open(mentions_file_path, 'r') as mentions_file:
        reader = csv.DictReader(mentions_file)

        for row in reader:
            movie_db_map[row['movieId']] = row['movieName']
    
    return movie_db_map

def try_load_pickle(pickle_file_path):
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)

def save_pickle(pickle_file_path, data):
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)

def prepare_redial_baseline_dataset(
    redial_path, 
    tokenizer, 
    movie_db_map,
    dataset_cache_path='dataset_cache.pkl'
):

    dataset = try_load_pickle(dataset_cache_path)

    if dataset:
        print("Cached data already found, returning")
        return dataset
    
    split_files = {
        'train': 'train_data.jsonl',
        'test': 'test_data.jsonl'
    }

    dataset = {}

    for split, split_file_name in split_files.items():
        split_file_path = os.path.join(redial_path, split_file_name)
        examples = prepare_baseline_redial_split(split_file_path, tokenizer, movie_db_map)

        dataset[split] = examples
    
    save_pickle(dataset_cache_path, dataset)
    print("Saved file to cache ", dataset_cache_path)
    return dataset
