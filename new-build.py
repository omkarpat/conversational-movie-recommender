"""
 Created by diesel
 11/9/20
"""





from lexicon import Lexicon, LexBuilder
from rule_based_ner import RuleBasedNER


import file_utils as fu
from shutil import copyfile

import unidecode

import os
import json
import nltk
from collections import Counter
from collections import defaultdict
import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')



def unicode_2_ascii(text):
    #if isinstance(text, unicode):
    text = unidecode.unidecode(text)
    return text

def iter_lines(lines):
    for line in lines:
        yield line



def unpack_entry(first, lines, val_lists=False):
    # first line is the etype
    #if not first:

    d = {
        "category": first.strip(" :").lower()
    }
    single_string_values = {"text", "follow_up", "followup", "follow_up", "tid", "evaluative_language",
                            "dact", "etype", "mod", "rel", "primary_dact", "uid"}


    #print("first:", first)
    #print("lines[0]:", lines[0])
    for line in lines:
        if line.startswith("#"):
            continue
        line = line.strip()
        # stop at first empty line
        if not line:
            break

        splitline = line.split(":", maxsplit=1)
        #print("splitline:", splitline)

        slot_name, slot_vals = splitline
        slot_vals = unicode_2_ascii(slot_vals)

        if (val_lists and slot_name not in single_string_values) or slot_name in {"reference_forms", "forms", "referential_expressions"}:
            if "," in slot_vals:
                slot_vals = [v.strip() for v in slot_vals.split(",") if v.strip()]
                #print("ifif:", slot_vals)
            else:
                slot_vals = [v.strip() for v in slot_vals.split(";") if v.strip()]
                #print("ifelse")
        else:
            slot_vals = slot_vals.strip()
            #print("else")
        slot_name = slot_name.strip().lower()
        if slot_name == "tid":
            slot_name = "uid"
        elif slot_name == "followup":
            slot_name = "follow_up"
        d[slot_name] = slot_vals
        #print(f"{slot_name}: {slot_vals}")

    return d, d["category"]


def extract_referencial_forms(e, cat, fullname_only=False):
    if cat == "person":
        names = [e.get(name) for name in ["first_name", "last_name", ] if e.get(name)]
        full_name = " ".join(names)
        forms = [full_name]
        if not fullname_only and e.get("last_name"):
            forms.append(e["last_name"])
        e["full_name"] = full_name

    elif cat == "book":
        forms = [e[attr] for attr in ["full_name", "nick_name" ] if e.get(attr)]
        if e.get("nick_name") and e["nick_name"].startswith("the "):
            forms.append(e["nick_name"][len("the "):].strip())
        _ref_field = "reference_forms"
        if e.get(_ref_field):
            if not isinstance(e[_ref_field], list):
                e[_ref_field] = [e[_ref_field]]
            forms.extend(e[_ref_field])
    else:
        forms = [e[name] for name in ["name", "nick_name", "full_name",] if e.get(name)]
        if e.get("forms"):
            forms.extend(e["forms"])
        forms = list(set(forms))
        if not e.get("full_name") and e.get("name"):
            e["full_name"] = e["name"]

    #print("e forms:", e["forms"])
    if forms:
        e["forms"] = forms
    return e


def demo():
    center_dir = "../center/astext-cleaned/"
    topic_name = "animals"
    #fact_infile_name = f"{topic_name}-facts.txt"
    lex_infile_name = f"{topic_name}-lex.txt"
    lex_input_path = os.path.join(center_dir, lex_infile_name)
    lines = fu.read_lines(lex_input_path)

    lines = iter_lines(lines)
    builder = LexBuilder()



    # entry is dictionary
    builder.add_entry({
        'category': 'animal',
        'standard_form': 'Cat',
        'forms': ['kittens', 'kitten', 'cats', 'feline', 'felines', 'kitty', 'kitty cats', 'kitty cat', 'Cat', 'kitties'],
        'name': 'Cat',
        'full_name': 'Cat'
    })
    builder.add_entry({
        'category': 'animal',
        'standard_form': 'Dog',
        'forms': ['Dog', 'canine', 'doggy', 'dogs', 'puppies', 'puppy'],
        'name': 'Dog',
        'full_name': 'Dog'
    })
    builder.add_entry({
        'category': 'concept',
        'standard_form': 'gender',
        'forms': ['gender'],
        'name': 'gender',
        'full_name': 'gender'
    })
    builder.add_entry({
        'category': 'concept',
        'standard_form': 'intelligence',
        'forms': ['learning', 'intelligence', 'cognitive ability'],
        'name': 'intelligence',
        'full_name': 'intelligence'
    })
    builder.add_entry({
        'category': 'concept',
        'standard_form': 'anatomy',
        'forms': ['anatomy'],
        'name': 'anatomy',
        'full_name': 'anatomy'
    })

    builder.build_lexes()
    builder.save_lexes(os.path.join("./", f"{topic_name}-lex.json"))

    print("entry keys:", builder.get_entry_key_set())

    ner_tagger = RuleBasedNER(builder.lexicons)

    texts = ["i think cats dog", "i think cats learning gender and anatomy", "i think cats learning",
             "this is cats weird and dogs"]
    for text in texts:
        print("\ntext:", text)
        toks = ner_tagger.tokenize_text(text)
        mentions = ner_tagger.tag_tokens(toks)
        print("mentions:")
        for m in mentions:
            print(" - ", m)

        print()




def main():
    center_dir = "../center/astext-cleaned/"
    topic_name = "hockey"
    topic_name = "animals"
    #topic_name = "activities"
    #topic_name = "harry_potter"
    #fact_infile_name = f"{topic_name}-facts.txt"
    lex_infile_name = f"{topic_name}-lex.txt"
    lex_input_path = os.path.join(center_dir, lex_infile_name)
    lines = fu.read_lines(lex_input_path)

    lines = iter_lines(lines)
    builder = LexBuilder()

    for j, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        entry, etype = unpack_entry(line, lines)
        # entry is dictionary
        entry = extract_referencial_forms(entry, etype, fullname_only=True)
        builder.add_entry(entry, etype)


    builder.build_lexes()
    builder.save_lexes(os.path.join("./", f"{topic_name}-lex.json"))

    print("entry keys:", builder.get_entry_key_set())

    ner_tagger = RuleBasedNER(builder.lexicons)

    texts = ["i think cats dog", "i think cats learning gender and anatomy", "i think cats learning",
             "this is cats weird and dogs"]
    for text in texts:
        print("\ntext:", text)
        toks = ner_tagger.tokenize_text(text)
        mentions = ner_tagger.tag_tokens(toks)
        print("mentions:")
        for m in mentions:
            print(" - ", m)

        print()






if __name__ == "__main__":
    #main()
    demo()
