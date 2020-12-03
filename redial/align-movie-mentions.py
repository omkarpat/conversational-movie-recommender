
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


def normalize_imbd_id(the_id, wanted_len):
    if not isinstance(the_id, str):
        the_id = str(the_id)
    if len(the_id) < wanted_len:
        # imdb movie id's are length 7 numeric identifiers with
        # leading zeroes when necessary.
        the_id = "0" * (wanted_len - len(the_id)) + the_id
    return the_id

def get_movie(mid, movie_base):
    return movie_base.get(normalize_imbd_id(mid, 7))

def get_person(pid, person_base):
    return person_base.get(normalize_imbd_id(pid, 6))

def main():

    with open("all-movies.json", "r") as fin:
        movies_db = json.load(fin)

    with open("all-people.json", "r") as fin:
        people_db = json.load(fin)

    for split_name in ["test", "train"]:
        new_lines = []
        fname = f"{split_name}_data_genre_tagged.jsonl"
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
                movie = get_movie(mid, movies_db)
                if isinstance(movie, dict):
                    for attr_name in ["director", "cast"]:
                        new_vals = [get_person(pid, people_db) if get_person(pid, people_db) else pid
                                    for pid in movie.get(attr_name, [])]

                        for p in new_vals:
                            if not isinstance(p, dict):
                                continue
                            new_kf_movies = []
                            for kf_movie_id in p.get("known for", []):
                                kf_movie = get_movie(kf_movie_id, movies_db)
                                if kf_movie is None:
                                    new_kf_movies.append(kf_movie_id)
                                else:
                                    new_kf_movies.append({
                                        "long imdb title": kf_movie.get("long imdb title", "_unk_title_"),
                                        "imdb_id": kf_movie_id
                                    })
                            p["known for"] = new_kf_movies

                        movie[attr_name] = new_vals

                    new_mentions[mid] = movie
                else:
                    # blindly copy whatever is there.
                    new_mentions[mid] = d["movieMentions"][mid]
            #print(new_mentions[mid])
            d["movieMentions"] = new_mentions
            new_lines.append(json.dumps(d))

        fname = f"{split_name}-with-movie-info.jsonl"
        print("saving", fname)
        with open(fname, "w") as fout:
            fout.write("\n".join(new_lines))


if __name__ == "__main__":
    main()



