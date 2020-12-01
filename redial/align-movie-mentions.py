
import pandas as pd
import numpy as np
import json
import nltk

def make_signature(row):
    return f"{row.original_title}  ({row.year})".lower()

def main():
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







if __name__ == "__main__":
    main()



