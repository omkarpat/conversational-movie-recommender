import requests
import json
import logging
import argparse
import os

import pdb
import re
from tqdm.auto import tqdm

from collections import defaultdict

multi_spaces_pattern = re.compile(r"\s+")

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

    # These correspond to labels that have been misrecognized
    blacklist_URIs = {"http://dbpedia.org/resource/Glossary_of_tennis_terms", 
    "http://dbpedia.org/resource/Good_Movie",
    "http://dbpedia.org/resource/Sierra_Entertainment",
    "http://dbpedia.org/resource/Nice",
    "http://dbpedia.org/resource/Take_Care_(album)",
    "http://dbpedia.org/resource/Cloning",
    "http://dbpedia.org/resource/Blood",
    "http://dbpedia.org/resource/Downhill_creep",
    "http://dbpedia.org/resource/Movies",
    "http://dbpedia.org/resource/Hey_There",
    "http://dbpedia.org/resource/Swimming_(sport)",
    "http://dbpedia.org/resource/Princess_Falls",
    "http://dbpedia.org/resource/Haha_(entertainer)",
    "http://dbpedia.org/resource/LOL",
    "http://dbpedia.org/resource/Drag_queen",
    "http://dbpedia.org/resource/Yea_Football_Club",
    "http://dbpedia.org/resource/Oh_Yeah_(Yello_song)",
    "http://dbpedia.org/resource/Scalable_Coherent_Interface",
    "http://dbpedia.org/resource/CAN_bus",
    "http://dbpedia.org/resource/The_New_One_(horse)",
    "http://dbpedia.org/resource/Information_technology",
    "http://dbpedia.org/resource/The_Glad_Products_Company",
    "http://dbpedia.org/resource/AM_broadcasting",
    "http://dbpedia.org/resource/To_Heart",
    "http://dbpedia.org/resource/National_Organization_for_Women",
    "http://dbpedia.org/resource/Hit_or_Miss_(New_Found_Glory_song)",
    "http://dbpedia.org/resource/Canada",
    "http://dbpedia.org/resource/Different_Things",
    "http://dbpedia.org/resource/Norwegian_Trekking_Association",
    "http://dbpedia.org/resource/Take_One_(Canadian_magazine)",
    "http://dbpedia.org/resource/For_Inspiration_and_Recognition_of_Science_and_Technology",
    "http://dbpedia.org/resource/Two_Guys",
    "http://dbpedia.org/resource/The_Sydney_Morning_Herald",
    "http://dbpedia.org/resource/Booting",
    "http://dbpedia.org/resource/Precious_Time_(album)",
    "http://dbpedia.org/resource/I\\u0027m_Glad",
    "http://dbpedia.org/resource/Social_Democratic_Party_of_Switzerland",
    "http://dbpedia.org/resource/International_Maritime_Organization",
    "http://dbpedia.org/resource/LOL",
    "http://dbpedia.org/resource/Names_of_God_in_Judaism",
    "http://dbpedia.org/resource/Ike_Turner",
    "http://dbpedia.org/resource/Tricky_Stewart",
    "http://dbpedia.org/resource/Movies!",
    }

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


            if spotlight_annotations.get("Resources"):
                for entity in spotlight_annotations["Resources"]:
                    if entity["@URI"] in blacklist_URIs:
                        break
                    if entity["@types"]:
                        for t in entity["@types"].split(","):
                            recognized_category_types[t] += 1
                # else:
                    # if entity["@URI"] not in categories_to_not_stop_at:                   
                    #     pdb.set_trace()
                cleaned_mentions = [entity for entity in spotlight_annotations["Resources"] if entity["@URI"] not in blacklist_URIs]
                message["spotlight_mentions"] = cleaned_mentions
    return conversations
    

def process_text(text):
    return multi_spaces_pattern.sub(" ", text.capitalize())

def load_conversations(conversations_file):
    with open(conversations_file, 'r') as conversations_file:
        conversations = []

        for line in conversations_file:
            conversations.append(json.loads(line.strip()))
    
        return conversations

def dump_conversations_to_file(conversations, output_file):
    with open(output_file, 'w') as conversations_file:
        for conversation in conversations:
            conversations_file.write(json.dumps(conversation))
            conversations_file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path', 
        type=str,
        default="redial"
    )

    args = parser.parse_args()

    splits = {
        "train": {"input_file": "train_data.jsonl", "output_file": "train_data_spotlight.jsonl"},
        "test": {"input_file": "test_data.jsonl", "output_file": "test_data_spotlight.jsonl"}
    }
    
    for split, metadata in splits.items():

        split_filepath = os.path.join(args.dataset_path, metadata["input_file"])

        conversations = load_conversations(split_filepath)

        annotated_conversations = spotlight_annotate_conversations(conversations)

        out_filepath = os.path.join(args.dataset_path, metadata["output_file"])
        dump_conversations_to_file(annotated_conversations, out_filepath)