"""
 Created by diesel
 12/23/19
"""

#import scaffold
#print(vars(scaffold))



from lexicon import Lexicon
import json
import nltk
from collections import Counter


"""

        # Penn Tagset
        # Optional DT: (\S+/DT\s*)?
        # Adjetive can be JJ,JJR,JJS: (\S+/JJ\w?\s*)*
        # Noun can be NN,NNS,NNP,NNPS: (\S+/NN\w*\s*)+
        # ?P<> named group
        # ?: non-capturing group
        # See https://docs.python.org/3/howto/regex.html#non-capturing-and-named-groups

        # index/word/pos/ner

        # capital determiner 'A' or 'The' followed by nouns
        # The man
        # (The Markell family) always
        generic = "((?:(?:[0-9]+)/(?:The|A)/DT/\S+\s)(?:\S+/NN\S*/\S+\s*)+)"

        # possesive relations people
        #  our son; his girlfriend
        generic2 = "((?:(?:[0-9]+)/\S+/PRP\$/\S+\s)(?:\S+/NN\S*/\S+\s*)+)"

        # Mister Deals saw
        proper = "((?:^(?:[0-9]+)/\S+/NN\S*/\S+\s)(?:\S+/NN\S*/\S+)*)"

        person = "((?:(?:[0-9]+)/\S+/NN\S*/\S+\s+)*(?:\S+/NN\S*/PERSON\s*))"

        speaker = "((?:[0-9]+/SPEAKER/\S+/\S+\s+))"

        #char_patterns = "|".join([generic, proper, person, speaker])

        re_patterns = {
            "animate_NN_wn": generic,
            "proper_NN": proper,
            "person_NER": person,
            "speaker": speaker,
            "poss_rel_wn": generic2
        }


        #pattern = re.compile(r'(?P<CHAR>(?:\S+/DT\s*)?(?:\S+/JJ\w?\s*)*(?:\S+/NN\w*\s*)+)')
        #pattern = re.compile("(?P<CHAR>{})".format(char_patterns))

        regex = {k: re.compile("(?P<CHAR>{})".format(char_patterns)) for k, char_patterns in re_patterns.items()}

        #found_entities = defaultdict(list)
        for pat_name, pattern in regex.items():
            #print("pattern_name:", pat_name)
            for m in pattern.finditer(tokens):
                mention = [feats.split("/") for feats in m.group("CHAR").rstrip().split()]
"""



class RuleBasedNER():
    def __init__(self, lexicons):
        self._lexicons = lexicons

    def tokenize_text(self, text):
        return [t.lower() for t in nltk.word_tokenize(text)]

    def tag_tokens(self, tokens):
        #print("tokens:", tokens)

        indexed_grams = {}
        for n in [4, 3, 2, 1]:
            ngrams = nltk.ngrams(tokens, n)

            for j, gram in enumerate(ngrams):
                g = " ".join(gram)
                if g in indexed_grams:
                    if not isinstance(indexed_grams[g], list):
                        indexed_grams[g] = [indexed_grams[g]]
                    indexed_grams[g].append(j)
                else:
                    indexed_grams[g] = j

        #print("all_grams:", list(indexed_grams.keys()))

        mentions = []
        _mention_idxs = set()
        for lex_name, lex in self._lexicons.items():

            #print("phrases:", lex.phrases)

            found = set(indexed_grams.keys()) & lex.phrases
            #print("found:", found)
            found = sorted([(len(f),f) for f in found], reverse=True)

            if len(found) > 0:
                for length, gram in found:
                    #print("gram:", gram)
                    idxs = indexed_grams.get(gram)
                    #print("idxs:", idxs)
                    if not isinstance(idxs, list):
                        idxs = [idxs]

                    for start_idx in idxs:
                        #print("start_idx:", start_idx)
                        #print("len:", len(gram.split()))
                        end_idx = start_idx + len(gram.split())

                        span_idxs = set(range(start_idx, end_idx))
                        #print("span_idxs:", span_idxs)
                        #print("start,end:", start_idx, end_idx)
                        standard_form, ent_id = lex.ref2standard(gram, get_tid=True)
                        #print("_mention_idxs:", _mention_idxs)
                        if len(_mention_idxs & span_idxs) < 1:
                            mention = {
                                "ent_id": ent_id,
                                "standard_form": standard_form,
                                "words": tokens[start_idx:end_idx],
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "ent_type": lex.name
                            }
                            mentions.append(mention)
                            for _idx_ in span_idxs:
                                _mention_idxs.add(_idx_)

        #print("mentions:", mentions)
        return mentions



def player_analysis():
    dm = DataManager(
        season_dir_path="../data/season_info",
        players_path="../data/nba-static-data/nba-players.json",
        teams_path="../data/nba-static-data/nba-teams.json",
        fun_facts_path="../data/ready-facts.json",
        questions_path="../data/ready-questions.json",
        standings_path="../data/nba-playoff-picture",
    )

    print("All Players:")
    full_names = [player["full_name"] for player in dm.players]
    print("num_names:", len(full_names))
    print("num unique names:", len(set(full_names)))

    dups = []
    _all = set()
    for name in full_names:
        if name in _all:
            dups.append(name)
        _all.add(name)

    for d in dups:
        print(" * ", d)

    print("Active Players:")
    full_names = [player["full_name"] for player in dm.players if player["is_active"]]
    print("num_names:", len(full_names))
    print("num unique names:", len(set(full_names)))


def process_ref(text):
    return " ".join(nltk.word_tokenize(text)).lower()


def setup_lexicons():
    dm = DataManager(
        season_dir_path="../data/season_info",
        players_path="../data/nba-static-data/nba-players.json",
        teams_path="../data/nba-static-data/nba-teams.json",
        fun_facts_path="../data/ready-facts.json",
        questions_path="../data/ready-questions.json",
        standings_path="../data/nba-playoff-picture",
    )


    # set up lexicons of know entities
    player_lex = Lexicon("player")
    for player in dm.players:
        refs = [process_ref(player["full_name"])]
        player_lex.new_term(player["full_name"], info=player, referential_forms=refs)

    team_lex = Lexicon("team")
    for team in dm.teams:
        refs = [process_ref(team[k]) for k in ["full_name", "city", "nickname", ]] # "abbreviation"
        team_lex.new_term(team["full_name"], info=team, referential_forms=refs)

    lexicons = {
        lex.name: lex.to_dict() for lex in [player_lex, team_lex]
    }

    with open("../data/lexicons/lex.json", "w") as fout:
        json.dump(lexicons, fout, indent=2)

    with open("../data/lexicons/lex.json", "r") as fin:
        lexicons2 = json.load(fin)

    lex2 = {
        name: Lexicon.from_dict(d) for name, d in lexicons2.items()
    }

    prepared_lexicons = "../data/lexicons/lex.json"
    with open(prepared_lexicons, "r") as fin:
        lexicons2 = json.load(fin)

    lexicons = {
        name: Lexicon.from_dict(d) for name, d in lexicons2.items()
    }


def team_ner_demo(fact, ner_tagger, results, lex):
    print(("-" * 20) + "\n\n")
    text = fact["text"]
    team = fact["team"]
    toks = [t.lower() for t in nltk.word_tokenize(text)]
    mentions = ner_tagger.tag_tokens(toks)
    #player_mentions = [men for men in mentions if men["ent_type"] == "player"]
    team_mentions = [men for men in mentions if men["ent_type"] == "team"]

    nicknames = [lex.get_info(men["standard_form"][0])["nickname"] for men in team_mentions]
    nicknames = "; ".join(nicknames)

    print("\ntext:", text)
    print("fact team:", team)
    print("team_mention:", "; ".join([" ".join(men["words"]) for men in team_mentions]))
    print("nickname:", nicknames)

    if len(team_mentions) == 1:
        words = " ".join(team_mentions[0]["words"])
        standard = team_mentions[0]["standard_form"][0]
    else:
        words = "; ".join([" ".join(men["words"]) for men in team_mentions])
        standard = "; ".join([men["standard_form"][0] for men in team_mentions])
    #print("player_mention:   ", words)
    print("{} == {}".format(team, nicknames))
    if team.lower() == nicknames.lower():
        result = "PASS"
    else:
        result = "FAIL"
    print(result)
    results.append(result)

def player_ner_demo(fact, ner_tagger, results):
    print(("-" * 20) + "\n\n")
    text = fact["text"]
    player = fact["player"]
    toks = [t.lower() for t in nltk.word_tokenize(text)]
    mentions = ner_tagger.tag_tokens(toks)
    player_mentions = [men for men in mentions if men["ent_type"] == "player"]
    team_mentions = [men for men in mentions if men["ent_type"] == "team"]
    print("\ntext:", text)
    print("player:", player)
    print("team_mention:", "; ".join([" ".join(men["words"]) for men in team_mentions]))
    if len(player_mentions) == 1:
        words = " ".join(player_mentions[0]["words"])
        standard = player_mentions[0]["standard_form"][0]
    else:
        words = "; ".join([" ".join(men["words"]) for men in player_mentions])
        standard = "; ".join([men["standard_form"][0] for men in player_mentions])
    print("player_mention:   ", words)
    print("{} == {}".format(player, standard))
    if player.lower() == standard.lower():
        result = "PASS"
    else:
        result = "FAIL"
    print(result)
    results.append(result)


def load_lexicons(prepared_lexicons="../data/lexicons/lex.json"):

    with open(prepared_lexicons, "r") as fin:
        lexicons = json.load(fin)

    lexicons = {
        name: Lexicon.from_dict(d) for name, d in lexicons.items()
    }

    return lexicons



def main():
    dm = DataManager(
        season_dir_path="../data/season_info",
        players_path="../data/nba-static-data/nba-players.json",
        teams_path="../data/nba-static-data/nba-teams.json",
        fun_facts_path="../data/ready-facts.json",
        questions_path="../data/ready-questions.json",
        standings_path="../data/nba-playoff-picture",
        templates_path="../data/season-templates/templates.json"
    )


    aka_list = [
        {"full_name": "Kareem Abdul-Jabbar", "aka": ["Kareem Abdul-Jabaar", "Kareem Abdul-Jabbar"]},
        {"full_name": "Shaquille O'Neal", "aka": ["Shaq"]}
    ]


    player_lex = Lexicon("player")
    for player in dm.players:
        refs = [process_ref(player["full_name"])]
        player_lex.new_term(player["full_name"], info=player, referential_forms=refs)

    for aka in aka_list:
        player_lex.new_referential_phrase(
            [process_ref(ref) for ref in aka["aka"]],
            aka["full_name"])


    team_lex = Lexicon("team")
    for team in dm.teams:
        refs = [process_ref(team[k]) for k in ["full_name", "city", "nickname"]] # "abbreviation"
        team_lex.new_term(team["full_name"], info=team, referential_forms=refs)

    lexicons = {
        lex.name: lex for lex in [player_lex, team_lex]
    }

    ner_tagger = RuleBasedNER(lexicons)

    results = []

    for fact in dm.fun_facts["fun-facts"]["players"]["facts"]:
        player_ner_demo(fact, ner_tagger, results)

    results = Counter(results)
    print("results:")
    for k,v in results.items():
        print("{}: {}".format(k, v))
    print("accuracy:", results["PASS"]/ (results["PASS"] + results["FAIL"]))


    results = []

    for fact in dm.fun_facts["fun-facts"]["teams"]["facts"]:
        team_ner_demo(fact, ner_tagger, results, lexicons["team"])

    results = Counter(results)
    print("results:")
    for k,v in results.items():
        print("{}: {}".format(k, v))
    print("accuracy:", results["PASS"]/ (results["PASS"] + results["FAIL"]))

    prepared_lexicons = "../data/lexicons/lex.json"
    lexicons = {name: lex.to_dict() for name, lex in lexicons.items()}
    with open(prepared_lexicons, "w") as fout:
        json.dump(lexicons, fout, indent=2)


if __name__ == "__main__":
    main()
