"""
 Created by diesel
 11/9/20
"""


import pandas as pd
import json


d = {
    'movieMentions': {
        '111776': 'Super Troopers (2001)',
        '91481': 'Beverly Hills Cop (1984)',
        '151656': 'Police Academy  (1984)',
        '134643': 'American Pie  (1999)',
        '192131': 'American Pie ',
        '124771': '48 Hrs. (1982)',
        '94688': 'Police Academy 2: Their First Assignment (1985)',
        '101794': 'Lethal Weapon (1987)'
    },
    'respondentQuestions': {
        '111776': {
            'suggested': 0,
            'seen': 1,
            'liked': 1
        },
        '91481': {
            'suggested': 1,
            'seen': 2,
            'liked': 2
        },
        '151656': {
            'suggested': 1,
            'seen': 0,
            'liked': 1
        },
        '134643': {
            'suggested': 0,
            'seen': 1,
            'liked': 1
        },
        '192131': {
            'suggested': 0,
            'seen': 1,
            'liked': 1
        },
        '124771': {
            'suggested': 1,
            'seen': 2,
            'liked': 2
        }, '94688': {
            'suggested': 1,
            'seen': 0,
            'liked': 1
        },
        '101794': {
            'suggested': 1,
            'seen': 0,
            'liked': 2
        }
    },
    'messages': [
        {
            'timeOffset': 0,
            'text': 'Hi I am looking for a movie like @111776',
             'senderWorkerId': 956,
            'messageId': 204171
        },
        {
            'timeOffset': 48,
            'text': 'You should watch @151656',
            'senderWorkerId': 957,
            'messageId': 204172
        },
        {
            'timeOffset': 90,
            'text': 'Is that a great one? I have never seen it. I have seen @192131',
            'senderWorkerId': 956,
            'messageId': 204173
        },
        {
            'timeOffset': 122,
            'text': 'I mean @134643',
            'senderWorkerId': 956,
            'messageId': 204174
        },
        {
            'timeOffset': 180,
            'text': 'Yes @151656 is very funny and so is @94688',
            'senderWorkerId': 957,
            'messageId': 204175
        },
        {
            'timeOffset': 199,
            'text': 'It sounds like I need to check them out',
            'senderWorkerId': 956,
            'messageId': 204176
        },
        {
            'timeOffset': 219,
            'text': 'yes you will enjoy them',
            'senderWorkerId': 957,
            'messageId': 204177
        },
        {
            'timeOffset': 253,
            'text': 'I appreciate your time. I will need to check those out. Are there any others you would recommend?',
            'senderWorkerId': 956,
            'messageId': 204178
        },
        {
            'timeOffset': 297,
            'text': 'yes @101794',
             'senderWorkerId': 957,
            'messageId': 204179
        },
        {
            'timeOffset': 311,
            'text': 'Thank you i will watch that too',
             'senderWorkerId': 956,
            'messageId': 204180
        },
        {
            'timeOffset': 312,
            'text': 'and also @91481',
             'senderWorkerId': 957,
            'messageId': 204181
        },
        {
            'timeOffset': 326,
            'text': 'Thanks for the suggestions.',
             'senderWorkerId': 956,
            'messageId': 204182
        },
        {
            'timeOffset': 341,
            'text': 'you are welcome',
             'senderWorkerId': 957,
            'messageId': 204183
        },
        {
            'timeOffset': 408,
            'text': 'and also @124771',
             'senderWorkerId': 957,
            'messageId': 204184
        },
        {
            'timeOffset': 518,
            'text': 'thanks goodbye',
             'senderWorkerId': 956,
            'messageId': 204185
        }
    ],
    'conversationId': '20001',
    'respondentWorkerId': 957,
    'initiatorWorkerId': 956,
    'initiatorQuestions': {
        '111776': {
            'suggested': 0, 'seen': 1, 'liked': 1},
        '91481': {
            'suggested': 1, 'seen': 2, 'liked': 2},
        '151656': {
            'suggested': 1, 'seen': 0, 'liked': 1},
        '134643': {
            'suggested': 0, 'seen': 1, 'liked': 1},
        '192131': {
            'suggested': 0, 'seen': 1, 'liked': 1},
        '124771': {
            'suggested': 1, 'seen': 2, 'liked': 2},
        '94688': {
            'suggested': 1, 'seen': 0, 'liked': 1},
        '101794': {
            'suggested': 0, 'seen': 2, 'liked': 2}}}



def get_messages(infile):
    with open(infile, "r") as fin:
        messages = []
        for line in fin:
            d = json.loads(line)

            speaker_lookup = dict(zip(range(len("ABCDEF")), "ABCDEF"))
            speakers = {}
            sid = 0
            for m in d["messages"]:
                #assert m['senderWorkerId'] < 7
                if m['senderWorkerId'] not in speakers:
                    speakers[m['senderWorkerId']] = speaker_lookup[sid]
                    sid += 1

                messages.append({
                    "text": m["text"],
                    "speaker": speakers[m['senderWorkerId']]
                })
    return messages



class DataLoader(object):
    def __init__(self, args, infile=None):
        self.args = args
        self.infile = infile



    def load(self, infile=None):
        if infile is None:
            infile = self.infile
        else:
            self.infile = infile

        messages = get_messages(infile)
        return pd.DataFrame(messages)



from lexicon import Lexicon, LexBuilder
from rule_based_ner import RuleBasedNER
import file_utils as fu
import os
from collections import defaultdict

def load_ner_tagger(concept_path, lex_save_path):
    df = pd.read_csv(concept_path, sep="\t")
    builder = LexBuilder()

    key_phrases = defaultdict(list)
    for row in df.itertuples():
        key_phrases[row.group.strip()].append(row.phrase.strip())

    for standard_form, forms in key_phrases.items():
        # entry is dictionary
        builder.add_entry({
            "category": "movie_genre",
            "standard_form": standard_form,
            "forms": forms,
            "name": standard_form,
            "full_name": standard_form,
        })

    builder.build_lexes()
    #builder.save_lexes(os.path.join("./", f"{topic_name}-lex.json"))
    builder.save_lexes(lex_save_path)

    #print("entry keys:", builder.get_entry_key_set())

    return RuleBasedNER(builder.lexicons)





def main():

    infile = "redial/train_data.jsonl"

    loader = DataLoader(None)
    messages = loader.load(infile)

    ner_tagger = load_ner_tagger("gez/genre-phrases.tsv", "gez/movie-lex.json")

    mentions = []
    for row in messages.itertuples():
        #print("\ntext:", row.text)
        toks = ner_tagger.tokenize_text(row.text)
        mentions.append(json.dumps(ner_tagger.tag_tokens(toks)))

    messages["genre_mentions"] = mentions

    #df = pd.DataFrame(messages)
    #df.to_csv("redial/train_data.messages.tsv")
    messages.to_csv("redial/train-genre-mentions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
