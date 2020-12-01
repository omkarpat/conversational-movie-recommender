import csv

import requests
import json
import logging
import argparse
import os
import spacy
import imdb
import pdb

from data_loader import load_ner_tagger
from data_utils import load_conversations, dump_conversations_to_file, popular_actors_list, popular_directors_list

from annotation_utils import DBPedia, process_text

from fuzzywuzzy import fuzz, process

import pdb
from tqdm.auto import tqdm

from collections import defaultdict

nlp = spacy.load("en_core_web_lg")

def spotlight_annotate(text):
    """
    Annotate the spotlight text with the default arguments
    """

    payload = {
        "text": text
    }
    headers = {
        "Accept": "application/json"
    }

    response = requests.get(
        "http://localhost:2222/rest/annotate", 
        headers=headers,
        params=payload
        )

    # print(response.text)
    return response.json()

def spotlight_annotate_conversations(conversations):

    categories_to_not_stop_at = {
        "http://dbpedia.org/resource/Romantic_comedy_film",
        "http://dbpedia.org/resource/Cartoon",
        "http://dbpedia.org/resource/Romance_film",
        "http://dbpedia.org/resource/Thriller_(genre)",
        "Lol! i am not a big tom fan at all."
    }

    # TODO : Use a secondary NER to identify mentions more accurately
    blacklist_types = {"Schema:MusicAlbum"}

    recognized_category_types = defaultdict(int)

    for conversation in tqdm(conversations):

        for message in conversation["messages"]:

            if not message["text"]:
                continue
            spotlight_annotations = spotlight_annotate(process_text(message["text"]))

            cleaned_mentions = []
            
            if spotlight_annotations.get("Resources"):
                for entity in spotlight_annotations["Resources"]:
                    if entity["@URI"] in DBPedia.BLACKLIST_URIS:
                        break
                    if entity["@types"]:
                        for t in entity["@types"].split(","):
                            recognized_category_types[t] += 1
                # else:
                    # if entity["@URI"] not in categories_to_not_stop_at:                   
                    #     pdb.set_trace()
                cleaned_mentions = [entity for entity in spotlight_annotations["Resources"] if entity["@URI"] not in DBPedia.BLACKLIST_URIS]
            
            message["spotlight_mentions"] = cleaned_mentions
    return conversations
    



def spacy_ner_annotate(conversations):
    recognized_category_types = defaultdict(int)

    for conversation in tqdm(conversations):
        for message in conversation["messages"]:
            doc = nlp(process_text(message["text"]))

            message_entities = []
            for ent in doc.ents:
                message_entities.append({
                    "surface": ent.text,
                    "type": ent.label_ 
                })

                recognized_category_types[ent.label_] += 1
            
            message["spacy_mentions"] = message_entities
    
    print(recognized_category_types)

    return conversations


def link_mention_to_imdb(conversations, imdb_sqlite_path=None):
    top_actors = popular_actors_list()
    top_directors = popular_directors_list()

    if imdb_sqlite_path:
        ia = imdb.IMDb('s3', os.path.join('sqlite+pysqlite:///', args.imdb_sqlite_path))
    else:
        ia = imdb.IMDb()

    match_cache = {}

    for conversation in tqdm(conversations):
        for message in conversation["messages"]:
            imdb_records = []

            if "spotlight_mentions" in message:

                for mention in message["spotlight_mentions"]:
                    if "Schema:Person" in mention["@types"]:
                        # Actor disambiguation logic
                        surface_form = mention["@surfaceForm"]

                        if surface_form == "Goodnight":
                            continue
                        
                        if surface_form.lower() == "jim" and "jim carrey" in message["text"].lower():
                            print("Jim converted to jim carrey")
                            surface_form = "Jim Carrey"

                        top_director_match, top_director_score = process.extractOne(surface_form, top_directors)

                        if top_director_score == 100:
                            # Tolerate only exact matches for director
                            print("Matched director ", top_director_match, "for ", surface_form)

                            
                            cached_result = match_cache.get(best_match)

                            if cached_result:
                                imdb_records.append(cached_result)
                            else:
                                imdb_person_results = ia.search_person(best_match)

                                if len(imdb_person_results) > 0:
                                    imdb_records.append(imdb_person_results[0])
                                    match_cache[best_match] = imdb_person_results[0]
                        else:
                            top_actor_matches = process.extract(surface_form, top_actors, limit=5, scorer=fuzz.partial_ratio)

                            num_high_matches = len([match for match, score in top_actor_matches if score > 80])

                            if num_high_matches > 1:
                                print("Warning! Multiple matches for ", surface_form, " ; Top matches: ", top_actor_matches)
                                num_100_matches = len([match for match, score in top_actor_matches if score == 100])

                                if num_100_matches == 1:
                                    
                                    best_match = top_actor_matches[0][0]
                                    print("Choosing closest match ", best_match)

                                    cached_result = match_cache.get(best_match)

                                    if cached_result:
                                        imdb_records.append(cached_result)
                                    else:
                                        imdb_person_results = ia.search_person(best_match)
                                        if len(imdb_person_results) > 0:
                                            imdb_records.append(imdb_person_results[0])
                                            match_cache[best_match] = imdb_person_results[0]
                                else:
                                    print("Too many high scoring matches!Ignoring!")
                                    print("Utterance: ", message["text"])
                                
                            elif num_high_matches == 0:
                                print("Warning! No match for surface form ", surface_form)
                                print("Utterance: ", message["text"])

                            else:
                                best_match = top_actor_matches[0][0]
                                cached_result = match_cache.get(best_match)

                                if cached_result:
                                    imdb_records.append(cached_result)
                                else:
                                    imdb_person_results = ia.search_person(best_match)
                                    if len(imdb_person_results) > 0:
                                        imdb_records.append(imdb_person_results[0])
                                        match_cache[best_match] = imdb_person_results[0]

            message["imdb_entries"] = [item.asXML() for item in imdb_records]
    return conversations

def load_movie_mentions_csv(mentions_csv_path):
    with open(mentions_csv_path, 'r') as csv_file:
        return [row for row in csv.DictReader(csv_file)]

def fix_movie_mentions(conversations, movie_corrected_mentions, imdb_sqlite_path):
    if imdb_sqlite_path:
        ia = imdb.IMDb('s3', os.path.join('sqlite+pysqlite:///', args.imdb_sqlite_path))
    else:
        ia = imdb.IMDb()

    def extract_imdb_id(url: str):
        return url.replace("https://www.imdb.com/title/tt", "").replace("/", "")

    for mention in movie_corrected_mentions:
        conv_id = mention["conversation_index"]
        message_index = mention["message_index"]

        conversation = conversations[int(conv_id)]

        message = conversation["messages"][int(message_index)]

        message_text = message["text"]

        movie_mentions_dict = conversation["movieMentions"]



        if mention["Movie Name 1"]:
            mention1 = mention["Movie Name 1"]
            imdb_id_1 = extract_imdb_id(mention["IMDB ID 1"])
            message_text = message_text.replace(mention1, imdb_id_1)

            movie_1 = ia.get_movie(imdb_id_1)
            movie_mentions_dict[imdb_id_1] = movie_1['title']


        if mention["Movie Name 2"]:
            mention2 = mention["Movie Name 2"]
            imdb_id_2 = extract_imdb_id(mention["IMDB ID 2"])

            movie_2 = ia.get_movie(imdb_id_2)
            movie_mentions_dict[imdb_id_2] = movie_2['title']

            message_text = message_text.replace(mention2, imdb_id_2)

        if mention["Movie Name 3"]:
            mention3 = mention["Movie Name 3"]
            imdb_id_3 = extract_imdb_id(mention["IMDB ID 3"])

            movie_3 = ia.get_movie(imdb_id_3)
            movie_mentions_dict[imdb_id_3] = movie_3['title']
            message_text = message_text.replace(mention3, imdb_id_3)

        message["text"] = message_text


        # Special corner case for correcting data
        if conv_id == "504" and message_index == "11" :
            message["text"] = message_text.replace("@Adam", "Adam")

    return conversations



def add_genre_mentions(conversations):
    ner_tagger = load_ner_tagger("gez/genre-phrases.tsv", "gez/movie-lex.json")

    for conversation in tqdm(conversations):
        for message in conversation["messages"]:
            tokenized_text = ner_tagger.tokenize_text(message["text"])

            mentions = ner_tagger.tag_tokens(tokenized_text)

            message["genre_mentions"] = mentions


    return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path', 
        type=str,
        default="redial"
    )
    parser.add_argument(
        '--imdb_sqlite_path',
        type=str,
        default=None,
        help="Path to the IMDB sqlite database. Optional, but makes search significantly faster"
    )

    args = parser.parse_args()

    splits = {
        "train": {"input_file": "train_data_imdb_corrected.jsonl", "output_file": "train_data_genre_tagged.jsonl"},
        "test": {"input_file": "test_data_imdb_corrected.jsonl", "output_file": "test_data_genre_tagged.jsonl"}
    }
    
    for split, metadata in splits.items():

        split_filepath = os.path.join(args.dataset_path, metadata["input_file"])

        conversations = load_conversations(split_filepath)

        annotated_conversations = add_genre_mentions(conversations)

        out_filepath = os.path.join(args.dataset_path, metadata["output_file"])
        dump_conversations_to_file(annotated_conversations, out_filepath)