import csv
import imdb
import pickle
import argparse

from collections import defaultdict
from tqdm.auto import tqdm

def retrieve_movies_data_from_imdb(args):
    """
    Given the merged movies data, retrieve data for ones which have a matching IMDB id
    """
    ia = imdb.IMDb()

    movies_data = defaultdict(dict)

    with open(args.merged_movie_data_path, 'r') as merged_movies_file:
        reader = csv.DictReader(merged_movies_file)
        for row in tqdm(reader):
            database_id = row['databaseId']
            imdb_id = row['imdbId']
            
            if database_id != '-1' and imdb_id != '-1': # We can definitively identify and retrieve these movies
                try:
                    movies_data['with_imdb_key'][database_id] = movie = ia.get_movie(imdb_id)
                    print("Processed movie:", movie)
                except imdb.IMDbError as e:
                    print("Exception", e)
                    movies_data['without_imdb_key'][database_id] = row
                    print("Skipped movie", row['movieName'])
            else: # Let's kick the can down the road by dealing with it later
                movies_data['without_imdb_key'][database_id] = row
                print("Skipped movie", row['movieName'])

    return movies_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_arg('--merged_movie_data_path', 
        default='redial/movies_merged_with_imdb.csv',
        type=str
    )

    args = parser.parse_args()

    movies_data = retrieve_movies_data_from_imdb(args)

    with open('imdb_data.pkl', 'wb') as movie_imdb_pickle_file:
        pickle.dump(movies_data, movie_imdb_pickle_file)