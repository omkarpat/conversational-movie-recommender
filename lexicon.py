"""
 Created by diesel
 12/23/19
"""

from collections import defaultdict

import nltk
import json

import spacy
nlp = spacy.load('en_core_web_sm')

class Lexicon(object):
    _save_attrs = ["name", "terms", "_id2term", "info", "_id_counter",
                   "_standard_2_id", "_all_refs"]
    def __init__(self, name):
        self.name = name
        self.terms = []
        self._id2term = {}
        self.info = {}
        self._id_counter = 0
        self._standard_2_id = {}
        self._all_refs = defaultdict(list)
        self._phrases = None


    def _clean_up(self):

        for record in self._id2term.values():
            record["referential_forms"] = list(set(record["referential_forms"]))


    def to_dict(self):
        self._clean_up()

        d = {}
        for k in self._save_attrs:
            d[k] = getattr(self, k)
        return d


    def to_lines(self , delim="; ", num_spaces=2):
        self._clean_up()

        all_lines = []
        for entry in self.terms:

            info = self.info[entry["id"]]

            category = self.name
            if category == "characters":
                category = "character"

            category = category.upper()

            lines = [category, "standard_form" + ": "+ entry["standard_form"]]
            forms = entry.get("referential_forms", [])
            if forms:
                forms = delim.join(forms)
                lines.append("forms: " + forms)

            for attr, val in info.items():
                if attr in {"forms", "full_name", "standard_form"}:
                    continue

                if isinstance(val, list):
                    val = delim.join(val)
                lines.append(attr + ": " + str(val))

            lines.extend([""] * num_spaces)
            all_lines.extend(lines)

        return all_lines


    def to_df(self , delim="; ", num_spaces=2):
        self._clean_up()

        all_lines = []
        for entry in self.terms:
            info = self.info[entry["id"]]
            category = self.name
            if category == "characters":
                category = "character"
            category = category.upper()
            e = {
                "category": category,
                "standard_form": entry["standard_form"],
                "forms": json.dumps(entry.get("referential_forms", []))
            }


            for attr, val in info.items():
                if attr in {"forms", "full_name", "standard_form"}:
                    continue
                if isinstance(val, list):
                    val = json.dumps(val)
                e[attr] = val



            all_lines.append(e)

        return all_lines




    @classmethod
    def from_dict(cls, d):
        assert "name" in d
        lex = cls(d["name"])
        for k in cls._save_attrs:
            if d.get(k):
                setattr(lex, k, d[k])

        lex._id2term = {}
        _terms = []
        if len(lex.terms) > 0:
            for t in lex.terms:
                if t["id"] not in lex._id2term:
                    lex._id2term[t["id"]] = t
                    _terms.append(t)

        lex.terms = _terms
        assert len(lex.terms) == len(lex._id2term), "{} {}".format(len(lex.terms), len(lex._id2term))

        return lex



    def get_info(self, tid_or_standard_form, default_val=None):
        info = self.info.get(tid_or_standard_form)
        if info is None:
            key = self._standard_2_id.get(tid_or_standard_form)
            info = self.info.get(key)

        if info is None:
            info = default_val
        return info


    def new_id(self):
        self._id_counter += 1
        return self._id_counter


    def new_term(self, standard_form, info=None, referential_forms=None, tid=None, standard_form_is_id=True):


        if standard_form in self._standard_2_id:
            the_id = self._standard_2_id[standard_form]
            for ref in referential_forms:
                self._add_new_ref(ref, the_id)

        else:
            new_term = {
                "standard_form": standard_form,
                "referential_forms": referential_forms if referential_forms else [],
                "id": (standard_form if standard_form_is_id else self.new_id()) if tid is None else tid
            }
            self.terms.append(new_term)
            self._id2term[new_term["id"]] = new_term
            self.info[new_term["id"]] = info

            for ref in referential_forms:
                self._add_new_ref(ref, new_term["id"])


            self._standard_2_id[standard_form] = new_term["id"]


    def update_info(self, tid, new_info):
        tinfo = self.info.get(tid, {})
        tinfo.update(new_info)


    def _add_new_ref(self, ref, tid):
        if ref not in self._all_refs or tid not in self._all_refs[ref]:
            if ref not in self._all_refs:
                self._all_refs[ref] = []
            self._all_refs[ref].append(tid)



    def new_referential_phrase(self, refs, standard_form=None, tid=None):
        assert standard_form or tid

        if tid is None:
            tid = standard_form

        if not isinstance(refs, list):
            refs = [refs]

        #print("tid:", [tid])
        record = self._id2term.get(tid)
        #print("record:", record)
        record["referential_forms"].extend([r for r in refs if r not in record["referential_forms"]])
        for ref in refs:
            self._add_new_ref(ref, tid)
            print(" * add {} => {}".format(ref, tid))






    @property
    def phrases(self):
        if self._phrases is None:
            self._phrases = set(self._all_refs.keys())

        return self._phrases


    def ref2standard(self, ref, get_tid=False):

        ids = self._all_refs.get(ref)
        if ids:
            if not isinstance(ids, list):
                ids = [ids]
            standard_forms = [self._id2term[_i].get("standard_form") for _i in ids]

        else:
            standard_forms = None

        retval = standard_forms, ids if get_tid else standard_forms
        return retval




class LexBuilder(object):
    def __init__(self):
        self._entries = defaultdict(list)
        self._lexicons = None

    @property
    def lexicons(self):
        return self._lexicons

    def add_entry(self, e, etype=None):
        if etype is None:
            etype = e["category"]
        self._entries[etype].append(e)

    def get_entry_key_set(self):
        return set([k for etype, ents in self._entries.items() for e in ents for k in e.keys()])

    def lex_as_dict(self):
        return {lex.name: lex.to_dict() for lex in self.lexicons.values()}


    def build_lexes(self):
        lexicons = {}
        for ent_type, entries in self._entries.items():
            lex = Lexicon(ent_type)
            for entry in entries:
                if isinstance(entry, str):
                    entry = entries[entry]
                if not entry.get("full_name") or entry["full_name"] == "none none":
                    continue
                refs = self.process_ref(entry["forms"])
                lex.new_term(entry["full_name"], info=entry, referential_forms=refs)
                #refs = [process_ref(player["full_name"])]
                #player_lex.new_term(player["full_name"], info=player, referential_forms=refs)
            lexicons[ent_type] = lex
        self._lexicons = lexicons


    def save_lexes(self, outfile):
        with open(outfile, "w") as fout:
            json.dump(self.lex_as_dict(), fout, indent=2)

    @staticmethod
    def process_ref(text):
        def _process(ref):
            doc = nlp(ref)
            toks = " ".join(getattr(tok, "text", None) for tok in doc).lower()
            #toks = " ".join(nltk.word_tokenize(ref)).lower()
            return toks
        retval = [_process(ref) for ref in text] if isinstance(text, list) else _process(text)
        return retval




def main():
    pass


if __name__ == "__main__":
    main()
