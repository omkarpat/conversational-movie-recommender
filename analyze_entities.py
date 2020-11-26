import json
import argparse
import re
import csv
import os

from tqdm.auto import tqdm

from data_utils import load_conversations

def identify_mislabeled_movies(conversations, output_file_path):
    mislabeled_movie_regex = re.compile(r"@[A-Za-z]")

    mislabeled_movie_utterances = []


    for i, conversation in enumerate(tqdm(conversations)):
        for j, message in enumerate(conversation["messages"]):
            
            match = mislabeled_movie_regex.search(message["text"], re.MULTILINE)

            if match:
                mislabeled_movie_utterances.append({
                    "conversation_index": i,
                    "message_index": j,
                    "messageId": message["messageId"],
                    "text": message["text"]
                })

    
    with open(output_file_path, 'w') as mislabeled_utterances_file:
        writer = csv.DictWriter(mislabeled_utterances_file, fieldnames=["conversation_index", "message_index", "text", "messageId"])

        writer.writeheader()
        writer.writerows(mislabeled_movie_utterances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')

    args = parser.parse_args()

    splits = {
        "train": {"input_file": "train_data_spacy.jsonl", "output_file": "train_utterances_with_mislabeled_movies.csv"},
        "test": {"input_file": "test_data_spacy.jsonl", "output_file": "test_utterances_with_mislabeled_movies.csv"}
    }

    for splitname, split_metadata in splits.items():

        input_file_path = os.path.join(args.data_path, split_metadata["input_file"])
        output_file_path = os.path.join(args.data_path, split_metadata["output_file"])
        conversations = load_conversations(input_file_path)

        identify_mislabeled_movies(conversations, output_file_path)