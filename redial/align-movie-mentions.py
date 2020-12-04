
import pandas as pd
import numpy as np
import json
import nltk

def make_signature(row):
    return f"{row.original_title}  ({row.year})".lower()

def main1():
    """
    ['Headhunter  (2009)']
    :return:
    """
    print("hello there !")

    redial_df = pd.read_csv("movies_with_mentions.csv")
    redial_movies = {row.movieName.lower(): j for j, row in enumerate(redial_df.itertuples())}

    imdb_df = pd.read_csv("IMDb-movies.csv")
    imdb_df["sig"] = [make_signature(row) for row in imdb_df.itertuples()]
    #to_drop = [row.Index for row in imdb_df.itertuples() if row.sig not in redial_movies]
    #imdb_df.drop(to_drop, axis=0, inplace=True)
    imdb_movies = {row.sig: (row.imdb_title_id, row.title ) for j, row in enumerate(imdb_df.itertuples())}

    mapping = [imdb_movies.get(row.movieName.lower()) for row in redial_df.itertuples()]
    redial_df["imdb_index"] = mapping

    redial_df.to_csv("movies_with_mentions-IMDb.tsv", sep="\t", index=False)
    imdb_df.to_csv("IMDb-movies-redial.tsv", sep="\t", index=False)


def read_lines(fpath):
    with open(fpath, "r") as fin:
        return fin.read().split("\n")

_long_count = 0
_good_count = 0
def normalize_imbd_id(the_id, wanted_len):
    global _long_count, _good_count
    if not isinstance(the_id, str):
        the_id = str(the_id)
    if len(the_id) < wanted_len:
        # imdb movie id's are length 7 numeric identifiers with
        # leading zeroes when necessary.
        the_id = "0" * (wanted_len - len(the_id)) + the_id
    #assert len(the_id) == wanted_len, [the_id]
    if len(the_id) != wanted_len:
        _long_count += 1
    else:
        _good_count += 1
    return the_id


def get_movie(mid, movie_base):
    return movie_base.get(normalize_imbd_id(mid, 7))

def get_person(pid, person_base):
    if isinstance(pid, str) or isinstance(pid, int):
        retval = person_base.get(normalize_imbd_id(pid, 7))
    else:
        retval = pid
    return retval

def main():

    movielense_2_imdb = {}
    database_2_imdb = {}
    df = pd.read_csv("movies_merged_with_imdb.csv")
    for row in df.itertuples():
        #print(row)
        imdbId = normalize_imbd_id(row.imdbId, 7)
        movielense_2_imdb[row.movielensId] = imdbId
        database_2_imdb[str(row.databaseId)] = imdbId
        assert isinstance(imdbId, str), imdbId

    _imdb_movie_set = set(movielense_2_imdb.values())
    assert "3874544" in _imdb_movie_set
    assert "0247745" in _imdb_movie_set
    assert database_2_imdb["111776"] == "0247745"

    movies_db, people_db = load_databases()
    #exit(0)
    for split_name in ["test", "train"]:
        new_lines = []
        fname = f"{split_name}_data_swda_tagged.jsonl"
        print("loading", fname)
        print("processing ...")
        for line in read_lines(fname):
            try:
                d = json.loads(line)
            except Exception as e:
                # copy the line over
                new_lines.append(line)
                # and continue to the next line
                continue

            new_mentions = {}
            for mid in d["movieMentions"]:

                #print("\nmid:", mid)
                imdb_id = database_2_imdb.get(mid)
                #print("imdb_id:", imdb_id)
                #print("mention:", d["movieMentions"][mid])
                movie = get_movie(imdb_id, movies_db)
                #print("movie:", movie)


                if isinstance(movie, dict):
                    for attr_name in ["director", "cast"]:
                        new_vals = [get_person(pid, people_db) if get_person(pid, people_db) else pid
                                    for pid in movie.get(attr_name, [])]

                        for p in new_vals:
                            if not isinstance(p, dict):
                                continue
                            new_kf_movies = []
                            for kf_movie_id in p.get("known for", []):
                                #print("kf_movie_id:", [kf_movie_id])
                                if isinstance(kf_movie_id, dict):
                                    kf_movie = kf_movie_id
                                else:
                                    kf_movie = get_movie(kf_movie_id, movies_db)

                                #print("kf_movie:", kf_movie)
                                if kf_movie is None:
                                    new_kf_movies.append(kf_movie_id)
                                else:
                                    new_kf_movies.append({
                                        "long imdb title": kf_movie.get("long imdb title", "_unk_title_"),
                                        "imdb_id": kf_movie_id.get("imdb_id") if isinstance(kf_movie_id, dict) else kf_movie_id
                                    })
                            p["known for"] = new_kf_movies
                        movie[attr_name] = new_vals
                    movie["imdb_id"] = imdb_id
                    new_mentions[mid] = movie

                else:
                    # blindly copy whatever is there.
                    new_mentions[mid] = d["movieMentions"][mid]
            #print(new_mentions[mid])
            d["movieMentions"] = new_mentions



            new_lines.append(json.dumps(d))

            if input(">>>"):
                exit(0)


        fname = f"{split_name}-with-movie-info.jsonl"
        print("saving", fname)
        with open(fname, "w") as fout:
            fout.write("\n".join(new_lines))

        print("_long_count:", _long_count)
        print("_good_count:", _good_count)


def load_databases():
    with open("all-movies.json", "r") as fin:
        movies_db = json.load(fin)
    mkeys = [len(str(mid)) for mid in list(movies_db.keys())]
    print("max movie key length:", max(mkeys))
    print("min movie key length:", min(mkeys))
    movies_db = {normalize_imbd_id(imdb_id, 7): entry
                   for imdb_id, entry in movies_db.items()}


    with open("all-people.json", "r") as fin:
        people_db = json.load(fin)

    pkeys = [len(str(mid)) for mid in list(movies_db.keys())]
    print("max people key length:", max(pkeys))
    print("min people key length:", min(pkeys))
    people_db = {normalize_imbd_id(imdb_id, 7): entry
                   for imdb_id, entry in people_db.items()}

    return movies_db, people_db


if __name__ == "__main__":
    main()



