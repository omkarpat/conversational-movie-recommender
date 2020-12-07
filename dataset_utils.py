import csv
import json
import os
import pickle

import re

import pdb
from bs4 import BeautifulSoup
from collections import namedtuple
from transformers import GPT2Tokenizer

from tqdm.auto import tqdm

BaselineExample = namedtuple(
    'BaselineExample',
    ['context', 'response']
)

KnowledgeGroundedExample = namedtuple(
    'KnowledgeGroundedExample',
    ['context', 'response', 'knowledge']
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

def prepare_redial_knowledge_grounded_split(
        split_path,
        movie_db_map,
        recommender_only=False,
        include_dact=True
):
    print("\nLoading data", split_path)

    with open(split_path, 'r') as split_file:
        split_conversations = split_file.read().splitlines()

    examples = []

    # Matching for movie mention ids: @1234
    movie_mention_pattern = re.compile(r"@(\d+)")
    # Pattern for mathching the year portion: (2007)
    movie_title_year_pattern = re.compile(r"\s+\(\d+\)")

    num_examples_using_knowledge = 0


    unk_terms = {
        "movie_genre": "<unk_genre>",
        "director": "<unk_director>",
        "cast": "<unk_cast>",
    }
    dact_set = set()

    for conversation_str in tqdm(split_conversations):
        conversation = json.loads(conversation_str)

        context = []

        messages = conversation["messages"]
        response = ""
        response_knowledge = []

        if recommender_only:
            sender_id = conversation["initiatorWorkerId"]

        for i, message in enumerate(messages):

            processed_text = message["text"]

            for mention in movie_mention_pattern.finditer(processed_text):
                movie_id = mention.group(1)

                movie_title = movie_db_map.get(movie_id)
                if not movie_title:
                    movie_title = conversation["movieMentions"][movie_id]
                    if isinstance(movie_title, dict):
                        movie_title = movie_title.get("title")

                movie_title = movie_title_year_pattern.sub('', movie_title)
                # naively substitute movie title in message
                processed_text = processed_text.replace("@" + movie_id, movie_title)

                response_knowledge.append(("movie_title", movie_title))
                #print("\n", movie_id)
                #print(conversation["movieMentions"].keys())
                mms = conversation["movieMentions"].get(movie_id)
                if isinstance(mms, dict):
                    mgenres = mms["genres"] if mms.get("genres") else [unk_terms["genre"]]
                    cast = [a["name"] + "," for a in mms["cast"]] if mms.get("cast") else [unk_terms["cast"]]
                    director = [a["name"] + "," for a in mms["director"]] if mms.get("director") else [unk_terms["director"]]

                    response_knowledge.extend([
                        ("movie_genre", " ".join(mgenres)),
                        ("director", " ".join(director)),
                        ("cast", " ".join(cast))
                    ])

            # For now, just pass in the surface form (later experiment is to try use normalized form)
            for genre_mention in message["genre_mentions"]:
                #print(genre_mention)
                response_knowledge.append(("genre", genre_mention["words"][0]))

            for imdb_entry in message["imdb_entries"]:
                soup = BeautifulSoup(imdb_entry, "xml")

                response_knowledge.append(('person', soup.find('name').text))


            if include_dact:
                dacts = []
                the_das = message["swbd_da"]
                for da in the_das:
                    d = "<" + da["label_name"].replace(" ", "_") + ">"
                    dact_set.add(d)
                    dacts.append(d)
                dact_tup = ("dact", " ".join(dacts))

                response_knowledge.append(dact_tup)
            #print(message)
            #print(response_knowledge)
            #input(">>")



            if i == len(messages) - 1 or \
                    message["senderWorkerId"] != messages[i + 1]["senderWorkerId"]:
                response += processed_text

                if not recommender_only or (recommender_only and sender_id != message["senderWorkerId"]):
                    examples.append(KnowledgeGroundedExample(
                        context,
                        response,
                        response_knowledge
                    ))

                    if len(response_knowledge) > 0:
                        num_examples_using_knowledge += 1

                context = context + [response]
                response = ""
                response_knowledge = []
            else:
                # We looked ahead and saw another follow-on response
                response += processed_text + " . "
    print("Num examples:", len(examples))
    print("Num examples using knowledge: ", num_examples_using_knowledge)
    return examples, list(unk_terms.values()) + list(dact_set)

def get_movie_db_map(mentions_file_path):
    movie_db_map = {}

    with open(mentions_file_path, 'r') as mentions_file:
        reader = csv.DictReader(mentions_file)

        for row in reader:
            movie_db_map[row['movieId']] = row['movieName']

    return movie_db_map

def try_load_pickle(pickle_file_path, get_special=False):
    print("trying to load pickle", pickle_file_path)
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)

        if get_special and isinstance(data, dict):
            retval = data["data"], data.get("special_terms")
        else:
            retval = data
        return retval
    print("not found ...")


def save_pickle(pickle_file_path, data, special_terms=None):

    if special_terms:
        data = {
            "data": data,
            "special_terms": special_terms
        }

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


def prepare_redial_knowledge_grounded_dataset(
    redial_path,
    tokenizer,
    movie_db_map,
    dataset_cache_path='kg_dataset_cache.pkl',
    split_files=None,
    recommender_only=False,
    include_dacts=True,
):
    dataset = try_load_pickle(dataset_cache_path, get_special=True)

    if dataset:
        print("Cached data already found, returning")
        return dataset[0], dataset[1]

    if split_files is None:
        split_files = {
            'train': 'train_data_genre_tagged.jsonl',
            'test': 'test_data_genre_tagged.jsonl'
        }

    dataset = {}

    for split, split_file_name in split_files.items():
        split_file_path = os.path.join(redial_path, split_file_name)
        examples, special_terms = prepare_redial_knowledge_grounded_split(split_file_path, movie_db_map, recommender_only, include_dacts)
        dataset[split] = examples
        if split.lower() == "train":
            train_terms = special_terms

    save_pickle(dataset_cache_path, dataset, special_terms=train_terms)
    print("Saved file to cache ", dataset_cache_path)
    return dataset, train_terms
