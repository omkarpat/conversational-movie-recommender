import csv
import imdb
import pickle
import argparse
import os
import requests
from collections import defaultdict
from tqdm.auto import tqdm, trange
from bs4 import BeautifulSoup


def retrieve_movies_data_from_imdb(args):
    """
    Given the merged movies data, retrieve data for ones which have a matching IMDB id
    """

    if args.imdb_sqlite_path:
        ia = imdb.IMDb('s3', os.path.join('sqlite+pysqlite:///', args.imdb_sqlite_path))
    else:
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

def scrape_imdb_list(url, n_pages):
    item_list = []

    index = 1

    for i in trange(1, n_pages + 1):
        formatted_url = url.format(str(i))

        response = requests.get(formatted_url)
        if response.status_code == 200:
            html = response.text

            soup = BeautifulSoup(html, 'html.parser')

            for header in soup.find_all('h3', "lister-item-header"):
                item_name = header.a.get_text().strip()

                item_struct = {
                    'id': index,
                    'name': item_name
                }

                item_list.append(item_struct)

                index += 1

    headers = ['id', 'name']

    return (item_list, headers)

def scrape_imdb_top_1000_actors():
    url = "https://www.imdb.com/list/ls058011111/?sort=list_order,asc&mode=detail&page={}"


    actor_list, headers = scrape_imdb_list(url, 10)

    with open('top_1000_actors.csv', 'w') as top_actors_file:
        writer = csv.DictWriter(top_actors_file, fieldnames=headers)

        writer.writeheader()
        writer.writerows(actor_list)
    

def scrape_imdb_top_250_directors():
    url = "https://www.imdb.com/list/ls008344500/"

    director_list, headers = scrape_imdb_list(url, 3)
    with open('top_250_directors.csv', 'w') as top_directors_file:
        writer = csv.DictWriter(top_directors_file, fieldnames=headers)

        writer.writeheader()
        writer.writerows(director_list)

def scrape_imdb_top_500_directors():
    url = "https://www.imdb.com/list/ls039888167/"

    director_list, headers = scrape_imdb_list(url, 5)
    with open('top_500_directors.csv', 'w') as top_directors_file:
        writer = csv.DictWriter(top_directors_file, fieldnames=headers)

        writer.writeheader()
        writer.writerows(director_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_movie_data_path', 
        default='redial/movies_merged_with_imdb.csv',
        type=str,
        help='Path to the merged file of imdb info and movielens data'
    )
    parser.add_argument('--imdb_sqlite_path',
        default='',
        type=str,
        help='Path to the IMDB sqlite data (for offline retrieval)'
    )

    args = parser.parse_args()

    # movies_data = retrieve_movies_data_from_imdb(args)

    # with open('imdb_data.pkl', 'wb') as movie_imdb_pickle_file:
    #     pickle.dump(movies_data, movie_imdb_pickle_file)

    # scrape_imdb_top_1000_actors()
    scrape_imdb_top_500_directors()