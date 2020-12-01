import pickle
from imdb import IMDb
import pandas as pd
from copy import deepcopy
import json

from imdb.Person import Person
from imdb.Movie import Movie
from imdb.Company import Company

from tqdm import tqdm


ia = IMDb()



def dump_person(p, max_n=None):
    """
    print("person:", p)
    print("current_info:", p.current_info)
    print("infoset2keys:", p.infoset2keys)
    d = dict(p)

    print(d)
    print("\n\n")
    for name, attr_set in p.infoset2keys.items():
        for attr in attr_set:
            print(f"{attr} = {p.get(attr)}")

    print("\n\n", "-"*20)
    #p = ia.get
    ia.update(p, info=["main", "biography", "awards"])

    print("person:", p)
    print("current_info:", p.current_info)
    print("infoset2keys:", p.infoset2keys)
    d = dict(p)

    print(d)
    print("\n\n")
    for name, attr_set in p.infoset2keys.items():
        print(name.upper())
        for attr in attr_set:
            r = p.get(attr)
            if isinstance(r, dict):
                r = r.keys()
            print(f"{attr} = {r}")
    """
    print("  * _person_", p)
    if p.myID not in all_people:
        ia.update(p, info=["main", "biography", "awards"])
        all_people[p.myID] = dump_item(dict(p), max_n)

    return p.myID

def dump_company(c, max_n=None):
    print("  * _company_", c)
    if c.myID not in all_companies:
        ia.update(c, info=["main",])
        all_companies[c.myID] = dump_item(dict(c), max_n)
    return c.myID

def dump_movie(m, max_n=None):
    print("  * _movie_", m)
    if m.movieID not in all_movies:
        all_movies[m.movieID] = True
        temp_d = dict(m)
        movie_d = {}
        for k, attr in temp_d.items():
            movie_d[k] = dump_item(attr, max_n)
        all_movies[m.movieID] = movie_d

    return m.movieID

def dump_item(val, max_n=None):

    if isinstance(val, list):
        trunc_val = len(val) if max_n is None else max_n
        dumped = [dump_item(v, max_n) for v in val[:trunc_val]]
    elif isinstance(val, dict):
        dumped = {k: dump_item(v, max_n) for k,v in val.items()}
    elif isinstance(val, Person):
        dumped = dump_person(val, max_n)
    elif isinstance(val, Movie):
        dumped = dump_movie(val, max_n)
    elif isinstance(val, Company):
        dumped = dump_company(val, max_n)
    else:
        # base case
        dumped = val
    return dumped



all_movies = {}
all_people = {}
all_companies = {}

def main():
    print("loading data ...")
    with open("imdb_data.pkl", "rb") as fin:
        imdb_data = pickle.load(fin)

    print("aggregating data ...")
    for k, movie in tqdm(list(imdb_data["with_imdb_key"].items())):
        all_movies[k] = dump_item(movie, 1)
        break

    print("saving data ...")
    for name, data in {"movies": all_movies, "people": all_people, "companies": all_companies}.items():
        with open(f"all-{name}.json", "w") as fout:
            json.dump(data, fout, indent=2)


def demo_main():
    #ia = IMDb()
    print("ia.get_movie_infoset():", ia.get_movie_infoset())
    print("ia.get_person_infoset():", ia.get_person_infoset())
    print("ia.get_company_infoset():", ia.get_company_infoset())
    exit(0)
    with open("imdb_data.pkl", "rb") as fin:
        d = pickle.load(fin)
    all_movies = {}
    all_people = {}
    for k, v in d["with_imdb_key"].items():
        # print(f"{k}: {type(v)}")
        print(f"{v.myTitle}:", v.current_info)
        print(v.infoset2keys)
        print(v.get("title"))
        print(v.get("taglines"))
        print("median" in v)

        ia.update(v, info=['taglines', 'vote details'])
        print(f"{v.myTitle}:", v.current_info)
        print(v.infoset2keys)
        print(v.get("title"))
        print(v.get("taglines"))
        print("median" in v)
        for a in ['demographics', 'number of votes', 'arithmetic mean', 'median']:
            print(f"{a}:", v.get(a))

        for kk, vv in v.items():
            # print(f"{kk}: {vv}")
            # if kk == "cast":
            if False:
                p = vv[0]
                print("-" * 10)
                print("cast[0]")
                print(p.summary())
                print(p.myName)
                print(p.personID)
                print(p.billingPos)
                for a in p.__dict__.keys():
                    print(f"{a}: {p.get(a)}")
                print("-" * 10)

                for attr_name in p.keys_alias:
                    print(f"{attr_name}: {p.get(attr_name)}")
                print("-" * 20)

                p = dict(name='', personID=p.personID, myName=p.myName,
                         myID=p.myID, data=p.data,
                         currentRole=p.currentRole,
                         roleIsPerson=p._roleIsPerson,
                         notes=p.notes, accessSystem=p.accessSystem,
                         titlesRefs=p.titlesRefs,
                         namesRefs=p.namesRefs,
                         charactersRefs=p.charactersRefs)
                for _k, _v in p.items():
                    print(f"{_k}: {_v}")

        _d = dict(v)
        print(_d)
        if input(">"):
            exit(0)
    df = pd.DataFrame([dict(movie) for movie in d["with_imdb_key"]])
    df.to_csv("imdb-movies.tsv", sep="\t", index=False)
    print("df shape:", df.shape)

if __name__ == "__main__":
    #demo_main()
    main()

