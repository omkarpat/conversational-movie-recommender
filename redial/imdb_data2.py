
import pickle
from imdb import IMDb
import pandas as pd
from copy import deepcopy
import json

from imdb.Person import Person
from imdb.Movie import Movie
from imdb.Company import Company

from tqdm.auto import tqdm, trange
from collections import defaultdict, Counter
import os


#movies_data = retrieve_movies_data_from_imdb(args)
ia = IMDb('s3', os.path.join('sqlite+pysqlite:///', "imdb.db"))
#ia = IMDb()
"""
{
    'data': {
        'original title': 'Headhunter (2009)',
        'cast': [<Person id:0586565[http] name:_Lars Mikkelsen_>, <Person id:0612675[http] name:_Charlotte Munck_>, <Person id:1649756[http] name:_Burkhard Forstreuter_>, <Person id:0816868[http] name:_Søren Spanning_>, <Person id:0275635[http] name:_Charlotte Fich_>, <Person id:0527937[http] name:_Troels Lyby_>, <Person id:0409745[http] name:_Henrik Ipsen_>, <Person id:3596023[http] name:_Katrine Læssøe Agesen_>, <Person id:3553227[http] name:_August Igor Svideniouk Egholm_>, <Person id:0365209[http] name:_Preben Harris_>, <Person id:0605786[http] name:_Henning Moritzen_>, <Person id:1028861[http] name:_Anders Valentinus Dam_>, <Person id:0061055[http] name:_David Bateson_>, <Person id:0488822[http] name:_Kjeld Nørgaard_>, <Person id:0297076[http] name:_Samuel Fröler_>, <Person id:0383863[http] name:_Bendt Hildebrandt_>, <Person id:0256953[http] name:_Flemming Enevold_>, <Person id:1181149[http] name:_Jakob B. Engmann_>, <Person id:2880345[http] name:_Camilla Gottlieb_>, <Person id:3595164[http] name:_Esbjørn Mildh_>, <Person id:0421680[http] name:_Simon Vagn Jensen_>, <Person id:0062966[http] name:_Teis Bayer_>, <Person id:0919261[http] name:_Rikke Weissfeld_>, <Person id:3177919[http] name:_Lars Huniche_>, <Person id:0421316[http] name:_Andrea Vagn Jensen_>, <Person id:3596071[http] name:_Espen Uldal_>, <Person id:0368646[http] name:_Vibeke Hastrup_>, <Person id:3594967[http] name:_Tove Bornhøft_>, <Person id:0369165[http] name:_Morten Hauch-Fausbøll_>, <Person id:1101365[http] name:_Jens Andersen_>, <Person id:1410527[http] name:_Sami Darr_>, <Person id:0029511[http] name:_Rita Angela_>, <Person id:0439579[http] name:_Maria Stokholm_>, <Person id:0794690[http] name:_Fash Shodeinde_>, <Person id:1596612[http] name:_Jesper Steinmetz_>, <Person id:1837586[http] name:_Signe Skov_>, <Person id:5555388[http] name:_Morten Bjørn_>, <Person id:4849296[http] name:_Martin Boserup_>, <Person id:5370548[http] name:_Rasmus Elton_>, <Person id:3065306[http] name:_Majamae_>, <Person id:0110249[http] name:_Adam Brix_>, <Person id:3280567[http] name:_Helena Faurbye Løfgren_>, <Person id:1401052[http] name:_David Petersen_>, <Person id:2897056[http] name:_Jacob Regel_>],
    'genres': ['Thriller'],
'runtimes': ['108'], 'countries': ['Denmark'],
'country codes': ['dk'],
'language codes': ['da'],
'color info': ['Color'], 'aspect ratio': '2.35 : 1', 'sound mix': ['Dolby Digital'], 'certificates': ['Denmark:15', 'Norway:15::(VOD)', 'Singapore:PG'],
'original air date': '28 Aug 2009 (Denmark)', 'rating': 6.6, 'votes': 1861,
'cover url': 'https://m.media-amazon.com/images/M/MV5BODcxNDk2NjExMl5BMl5BanBnXkFtZTgwOTg5NTAzMTE@._V1_SY150_CR9,0,101,150_.jpg',
'imdbID': '1322315',
'plot outline': "Martin Vinge, (35), former notorious journalist, now successful headhunter with a complicated personal life, is in all confidentiality contacted by 85 year-old N.F. Sieger, S.E.O. of Denmark's largest shipping company and oil empire. Sieger hires Martin to find an alternative heir to the firm instead of his son, Daniel Sieger, who for a long time has been destined to take the company into the next era. Martin starts coming up with suitable names for the position, but discovers that he has actually been entangled in a larger impenetrable power game aimed at deciding what is really going to happen to the company; a brutal power struggle that puts an intense pressure on Martin and his private life and relationships.",
'languages': ['Danish'],
'title': 'Headhunter', 'year': 2009, 'kind': 'movie',
'directors': [<Person id:0358530[http] name:_Rumle Hammerich_>],
'writers': [<Person id:0358530[http] name:_Rumle Hammerich_>, <Person id:None[http] name:_None_>, <Person id:0761879[http] name:_Åke Sandgren_>],
'producers': [<Person id:0536385[http] name:_Kim Magnusson_>, <Person id:0676896[http] name:_Louise Birk Petersen_>, <Person id:1277152[http] name:_Keld Reinicke_>, <Person id:0761879[http] name:_Åke Sandgren_>],
'composers': [<Person id:0343844[http] name:_Jacob Groth_>], 'cinematographers': [<Person id:0491565[http] name:_Dan Laustsen_>],
'editors': [<Person id:0804773[http] name:_Camilla Skousen_>, <Person id:1048600[http] name:_Henrik Vincent Thiesen_>],
'editorial department': [<Person id:1756751[http] name:_Christian Bentsen_>, <Person id:0433734[http] name:_Michael Jørgensen_>, <Person id:1293514[http] name:_Liv Lynge_>, <Person id:1515270[http] name:_Rebekka Lønqvist_>, <Person id:1245372[http] name:_Lone Goldie Møller_>, <Person id:0630930[http] name:_Michael Frank Nielsen_>, <Person id:0678566[http] name:_Glennie Pettersson_>, <Person id:0886806[http] name:_Nanna van Elk_>],
'production designers': [<Person id:0210539[http] name:_Peter De Neergaard_>], 'set decorators': [<Person id:1857851[http] name:_Mathias Hassing_>],
'costume designers': [<Person id:0632905[http] name:_Louize Nissen_>],
'make up department': [<Person id:0414709[http] name:_Dorte Jacobsen_>, <Person id:0766611[http] name:_Anne Cathrine Sauerberg_>],
'assistant directors': [<Person id:2802638[http] name:_Jakob Balslev_>, <Person id:0145252[http] name:_Antony Castle_>],
'art department': [<Person id:0350417[http] name:_Tine Gybel_>, <Person id:3200223[http] name:_Adrian Oskar Hansen_>, <Person id:3200393[http] name:_Lasse Kjederqvist_>, <Person id:1362097[http] name:_Christina Lorentz_>],
'sound department': [<Person id:1247185[http] name:_Morten Groth Brandt_>, <Person id:0214783[http] name:_Morten Degnbol_>, <Person id:1038633[http] name:_Morten Bak Jensen_>, <Person id:0486248[http] name:_Martin Langenbach_>, <Person id:0658889[http] name:_Rune Palving_>],
'special effects': [<Person id:1706530[http] name:_Søren Skov Haraldsted_>, <Person id:2284750[http] name:_Søren Hvam_>, <Person id:1139914[http] name:_Christian Kitter_>, <Person id:3278119[http] name:_Jacob Sebastian Malm_>], 'visual effects': [<Person id:0391540[http] name:_Michael Holm_>, <Person id:2922210[http] name:_Martin Madsen_>, <Person id:2850868[http] name:_Christian Schwanenflügel_>], 'stunts': [<Person id:1650709[http] name:_Munir Avn_>, <Person id:0110249[http] name:_Adam Brix_>, <Person id:10409553[http] name:_Jacob Sebastian Malm Carlsen_>, <Person id:0351618[http] name:_Stig Günther_>],
'camera department': [<Person id:2125660[http] name:_Thea Slouborg Andersen_>, <Person id:0145252[http] name:_Antony Castle_>, <Person id:1266070[http] name:_Karsten Jacobsen_>, <Person id:0421612[http] name:_Michael Wils Jensen_>, <Person id:2887813[http] name:_Allan Køldal_>, <Person id:1529598[http] name:_Josef Lopez_>, <Person id:1750238[http] name:_Magnus Ohmsen_>, <Person id:0711352[http] name:_John Frimann Rasmussen_>, <Person id:2699088[http] name:_Mathias Rasmussen_>, <Person id:3199386[http] name:_Claes Worm_>],
'casting department': [<Person id:1560324[http] name:_Tanja Grunwald_>], 'costume departmen': [<Person id:0260650[http] name:_Anne-Dorthe Eskildsen_>, <Person id:1787609[http] name:_Amalie Suurballe Schjoett-Wieth_>], 'location management': [<Person id:0840131[http] name:_Kim Scott Sutherland_>],
'music department': [<Person id:3446216[http] name:_Rasmus Bosse_>, <Person id:0471598[http] name:_Povl Kristian_>], 'script department': [<Person id:0100626[http] name:_Maj Bovin_>, <Person id:0292732[http] name:_Stine Monty Freddie_>],
'miscellaneous': [<Person id:2977486[http] name:_Signe Baasch_>, <Person id:2802638[http] name:_Jakob Balslev_>, <Person id:0188710[http] name:_Nina Crone_>, <Person id:3148878[http] name:_Henrik Edeltoft_>, <Person id:3198574[http] name:_Morten Frank_>, <Person id:1380253[http] name:_Lena Haugaard_>, <Person id:3198507[http] name:_Malene Jacobsen_>, <Person id:1380474[http] name:_Ada Kaalund_>, <Person id:2774381[http] name:_Mette Bang Kristensen_>, <Person id:2040257[http] name:_Stinna Lassen_>, <Person id:3017357[http] name:_Pia Madsen_>, <Person id:2022040[http] name:_Julie Rix Petersen_>, <Person id:1459824[http] name:_Marie Peuliche_>, <Person id:1382149[http] name:_Henrik Zein_>], 'thanks': [<Person id:0034366[http] name:_Jens Arentzen_>, <Person id:4004193[http] name:_Erik Bagger_>, <Person id:4003041[http] name:_Gert Binggeli_>, <Person id:4002683[http] name:_Jytte Dalsbo_>, <Person id:0676851[http] name:_Jes Dorph-Petersen_>, <Person id:1164435[http] name:_Anne Sofie Espersen_>, <Person id:1563441[http] name:_Martin Greis-Rosenthal_>, <Person id:3994793[http] name:_Kai Hammerich_>, <Person id:3994861[http] name:_Fritz Hansen_>, <Person id:4003197[http] name:_Klaus Eusebius Jakobsen_>, <Person id:1464034[http] name:_Karsten Jansfort_>, <Person id:0461339[http] name:_Joachim Knop_>, <Person id:1310618[http] name:_Peter Lund Madsen_>, <Person id:0540295[http] name:_Nils Malmros_>, <Person id:1599391[http] name:_Louis Poulsen_>, <Person id:4003162[http] name:_Camilla S._>, <Person id:3994926[http] name:_Nicolette Stoltze_>, <Person id:4003273[http] name:_Allan Strandlod_>, <Person id:4002637[http] name:_Heine Vangsgaard_>],
'akas': ['A fejvadász - A pénz hatalma (Hungary)', 'Ловец на глави (Bulgaria, Bulgarian title)', 'Doradztwo personalne (Poland)', 'Recrutorul (Romania)', 'Охотник за головами (Russia)'],
'writer': [<Person id:0358530[http] name:_Rumle Hammerich_>, <Person id:0761879[http] name:_Åke Sandgren_>], 'director': [<Person id:0358530[http] name:_Rumle Hammerich_>], 'production companies': [<Company id:0048411[http] name:_Det Danske Filminstitut_>, <Company id:0245119[http] name:_Nordisk Film Valby_>, <Company id:0054911[http] name:_TV2 Danmark_>], 'distributors': [<Company id:0213829[http] name:_Nordisk Film Distribution_>], 'special effects companies': [<Company id:0043208[http] name:_Dansk Speciel Effekt Service_>], 'other companies': [<Company id:0311084[http] name:_European Film Bonds_>, <Company id:0428163[http] name:_IMS Film & Media Insurance_>], 'plot': ["Martin Vinge, (35), former notorious journalist, now successful headhunter with a complicated personal life, is in all confidentiality contacted by 85 year-old N.F. Sieger, S.E.O. of Denmark's largest shipping company and oil empire. Sieger hires Martin to find an alternative heir to the firm instead of his son, Daniel Sieger, who for a long time has been destined to take the company into the next era. Martin starts coming up with suitable names for the position, but discovers that he has actually been entangled in a larger impenetrable power game aimed at deciding what is really going to happen to the company; a brutal power struggle that puts an intense pressure on Martin and his private life and relationships.::Nordisk Film"]}, 'myID': None, 'notes': '',
'titlesRefs': {
    'Headhunter': <Movie id:1322315[http] title:_Headhunter (None)_>,
    'Taglines': <Movie id:1322315[http] title:_Taglines (None)_>,
    'Synopsis': <Movie id:1322315[http] title:_Synopsis (None)_>,

'Plot Keywords': <Movie id:1322315[http] title:_Plot Keywords (None)_>,
'Parents Guide': <Movie id:1322315[http] title:_Parents Guide (None)_>,
'Plot Summary': <Movie id:1322315[http] title:_Plot Summary (None)_>, 'Full Cast and Crew': <Movie id:1322315[http] title:_Full Cast and Crew (None)_>, 'Release Dates': <Movie id:1322315[http] title:_Release Dates (None)_>, 'Official Sites': <Movie id:1322315[http] title:_Official Sites (None)_>, 'Company Credits': <Movie id:1322315[http] title:_Company Credits (None)_>, 'Filming & Production': <Movie id:1322315[http] title:_Filming & Production (None)_>, 'Technical Specs': <Movie id:1322315[http] title:_Technical Specs (None)_>, 'Trivia': <Movie id:1322315[http] title:_Trivia (None)_>, 'Goofs': <Movie id:1322315[http] title:_Goofs (None)_>, 'Crazy Credits': <Movie id:1322315[http] title:_Crazy Credits (None)_>, 'Quotes': <Movie id:1322315[http] title:_Quotes (None)_>, 'Alternate Versions': <Movie id:1322315[http] title:_Alternate Versions (None)_>, 'Connections': <Movie id:1322315[http] title:_Connections (None)_>,
'Soundtracks': <Movie id:1322315[http] title:_Soundtracks (None)_>, 'Photo Gallery': <Movie id:1322315[http] title:_Photo Gallery (None)_>, 'Trailers and Videos': <Movie id:1322315[http] title:_Trailers and Videos (None)_>, 'Awards': <Movie id:1322315[http] title:_Awards (None)_>, 'FAQ': <Movie id:1322315[http] title:_FAQ (None)_>, 'User Reviews': <Movie id:1322315[http] title:_User Reviews (None)_>, 'User Ratings': <Movie id:1322315[http] title:_User Ratings (None)_>, 'External Reviews': <Movie id:1322315[http] title:_External Reviews (None)_>, 'Metacritic Reviews': <Movie id:1322315[http] title:_Metacritic Reviews (None)_>, 'TV Schedule': <Movie id:1322315[http] title:_TV Schedule (None)_>, 'News': <Movie id:1322315[http] title:_News (None)_>, 'External Sites': <Movie id:1322315[http] title:_External Sites (None)_>}, 'namesRefs': {}, 'charactersRefs': {}, 'modFunct': <function modClearRefs at 0x7fd8e53a3510>, 'current_info': ['main', 'plot', 'synopsis'], 'infoset2keys': {'main': ['original title', 'cast', 'genres', 'runtimes', 'countries', 'country codes', 'language codes', 'color info', 'aspect ratio', 'sound mix', 'certificates', 'original air date', 'rating', 'votes', 'cover url', 'imdbID', 'plot outline', 'languages', 'title', 'year', 'kind', 'directors', 'writers', 'producers', 'composers', 'cinematographers', 'editors', 'editorial department', 'production designers', 'set decorators', 'costume designers', 'make up department', 'assistant directors', 'art department', 'sound department', 'special effects', 'visual effects', 'stunts', 'camera department', 'casting department', 'costume departmen', 'location management', 'music department', 'script department', 'miscellaneous', 'thanks', 'akas', 'writer', 'director', 'production companies', 'distributors', 'special effects companies', 'other companies'], 'plot': ['plot']}, 'key2infoset': {'original title': 'main', 'cast': 'main', 'genres': 'main', 'runtimes': 'main', 'countries': 'main', 'country codes': 'main', 'language codes': 'main', 'color info': 'main', 'aspect ratio': 'main', 'sound mix': 'main', 'certificates': 'main', 'original air date': 'main', 'rating': 'main', 'votes': 'main', 'cover url': 'main', 'imdbID': 'main', 'plot outline': 'main', 'languages': 'main', 'title': 'main', 'year': 'main', 'kind': 'main', 'directors': 'main', 'writers': 'main', 'producers': 'main', 'composers': 'main', 'cinematographers': 'main', 'editors': 'main', 'editorial department': 'main', 'production designers': 'main', 'set decorators': 'main', 'costume designers': 'main', 'make up department': 'main', 'assistant directors': 'main', 'art department': 'main', 'sound department': 'main', 'special effects': 'main', 'visual effects': 'main', 'stunts': 'main', 'camera department': 'main', 'casting department': 'main', 'costume departmen': 'main', 'location management': 'main', 'music department': 'main', 'script department': 'main', 'miscellaneous': 'main', 'thanks': 'main', 'akas': 'main', 'writer': 'main', 'director': 'main', 'production companies': 'main', 'distributors': 'main', 'special effects companies': 'main', 'other companies': 'main', 'plot': 'plot'}, '_Container__role': None, 'movieID': '1322315', 'myTitle': '', 'accessSystem': 'http',
'keys_tomodify': {'plot': None, 'trivia': None, 'alternate versions': None, 'goofs': None, 'quotes': None, 'dvd': None, 'laserdisc': None, 'news': None, 'soundtrack': None, 'crazy credits': None, 'business': None, 'supplements': None, 'video review': None, 'faqs': None
                  },
'_roleIsPerson': False, '_roleClass': <class 'imdb.Character.Character'>}

{'data': 
{
    'genres': ['thriller'], 'kind': 'movie', 'title': 'Headhunter', 
    'original title': 'Headhunter', 'adult': False, 'runtimes': [108], 
    'year': '2009', 
    'director': [<Person id:358530[s3] name:_Rumle Hammerich_>], 
'writer': [<Person id:761879[s3] name:_Åke Sandgren_>], 
'editor': [<Person id:804773[s3] name:_Camilla Skousen_>, <Person id:1048600[s3] name:_Henrik Vincent Thiesen_>], 
'cast': [<Person id:586565[s3] name:_Lars Mikkelsen_>, <Person id:612675[s3] name:_Charlotte Munck_>, <Person id:1649756[s3] name:_Burkhard Forstreuter_>, <Person id:816868[s3] name:_Søren Spanning_>], 
'composer': [<Person id:343844[s3] name:_Jacob Groth_>], 
'cinematographer': [<Person id:491565[s3] name:_Dan Laustsen_>], 
'rating': 6.6, 
'votes': 1858, 
'akas': [{'title': 'A fejvadász - A pénz hatalma', 'ordering': 1, 'types': ['imdbDisplay'], 'region': 'HU'}, {'title': 'Ловец на глави', 'ordering': 2, 'types': ['imdbDisplay'], 'language': 'bg', 'region': 'BG'}, {'title': 'Headhunter', 'ordering': 3, 'region': 'DK'}, {'title': 'Doradztwo personalne', 'ordering': 4, 'types': ['imdbDisplay'], 'region': 'PL'}, {'title': 'Recrutorul', 'ordering': 5, 'types': ['imdbDisplay'], 'region': 'RO'}, {'title': 'Headhunter', 'ordering': 6, 'types': ['imdbDisplay'], 'region': 'GB'}, {'title': 'Headhunter', 'ordering': 7, 'types': ['original'], 'original': True}, {'title': 'Охотник за головами', 'ordering': 8, 'types': ['imdbDisplay'], 'region': 'RU'}
         ]
}, 
'myID': None, 'notes': '', 'titlesRefs': {}, 'namesRefs': {}, 'charactersRefs': {}, 'modFunct': <function modClearRefs at 0x7f1e5292e510>, 'current_info': ['list', 'main', 'plot'], 'infoset2keys': {'main': ['genres', 'kind', 'title', 'original title', 'adult', 'runtimes', 'year', 'director', 'writer', 'editor', 'cast', 'composer', 'cinematographer', 'rating', 'votes', 'akas']}, 'key2infoset': {'genres': 'main', 'kind': 'main', 'title': 'main', 'original title': 'main', 'adult': 'main', 'runtimes': 'main', 'year': 'main', 'director': 'main', 'writer': 'main', 'editor': 'main', 'cast': 'main', 'composer': 'main', 'cinematographer': 'main', 'rating': 'main', 'votes': 'main', 'akas': 'main'}, '_Container__role': None, 'movieID': '1322315', 'myTitle': '', 'accessSystem': 's3', 'keys_tomodify': {'plot': None, 'trivia': None, 'alternate versions': None, 'goofs': None, 'quotes': None, 'dvd': None, 'laserdisc': None, 'news': None, 'soundtrack': None, 'crazy credits': None, 'business': None, 'supplements': None, 'video review': None, 'faqs': None}, 
'_roleIsPerson': False, '_roleClass': <class 'imdb.Character.Character'>}
"""
all_movies = {}
all_people = {}
all_companies = {}
freqs = defaultdict(list)

def dump_person(p, max_n=None):
    #print("  * _person_", p, p.personID) #, p.__dict__)
    #p = ia.get_person(p.personID)
    if p.myID not in all_people:
        # p = ia.get_person(p.personID)
        #ia.update(p, info=["main", "biography", "awards"])
        all_people[p.personID] = dump_item(dict(p), max_n)
    freqs["person"].append(p.personID)
    return p.personID

def dump_company(c, max_n=None):
    #print("  * _company_", c)
    if c.myID not in all_companies:
        #ia.update(c, info=["main",])
        all_companies[c.myID] = dump_item(dict(c), max_n)
    freqs["company"].append(c.companyID)
    return c.myID

def dump_movie(m, max_n=None, adding=False):
    #print("  * _movie_", m.__dict__)
    if m.movieID not in all_movies and adding:
        all_movies[m.movieID] = True
        temp_d = dict(m)
        movie_d = {}
        for k, attr in temp_d.items():
            movie_d[k] = dump_item(attr, max_n)
        all_movies[m.movieID] = movie_d

        #print("\n", movie_d)
        #exit(0)
    freqs["movies"].append(m.movieID)
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



def main():
    print("loading data ...")
    df = pd.read_csv("movies_merged_with_imdb.csv", dtype=str)
    df.drop("index", axis=1, inplace=True)
    print(df.head(10))
    df.drop([row.Index for row in df.itertuples() if int(row.imdbId) < 0], inplace=True)
    print(df.head(10))


    print("ia.get_movie_infoset():", ia.get_movie_infoset())

    print("aggregating data ...")
    pbar = tqdm(df.itertuples(), total=len(df))
    for row in pbar:
        #print("\nrow.movieName:", row.movieName, [row.imdbId])
        m = ia.get_movie(row.imdbId, info=["main"])
        #print("\n\n", m.current_info)
        #print(m.infoset2keys)
        #print("\n\ncast=", m['cast'])
        dump_movie(m, 4, adding=True)
        #pbar.set_description("Processing %s" % row.Index)
        #if len(all_movies) > 1:
        #    break
        #if row.Index > 100:
        #    break

    print("saving data ...")
    for name, data in {"movies": all_movies, "people": all_people, "companies": all_companies}.items():
        with open(f"all-{name}.json", "w") as fout:
            json.dump(data, fout, indent=2)

    all_freqs = []
    for t_name, entries in freqs.items():
        counts = Counter(entries)
        all_freqs.extend([{"record_type": t_name, "entry_name": e, "freq": n}
                          for e,n in counts.most_common()])

    all_freqs = pd.DataFrame(all_freqs)
    all_freqs.to_csv("all-freqs.tsv", sep="\t", index=False)
    print("all_freqs shape:", all_freqs.shape)

def get_imdb_people():
    print("loading data ...")
    with open(f"all-people.json", "r") as fin:
        d = json.load(fin)

    print("aggregating data ...")
    for pid in tqdm(d):
        p = ia.get_person(pid)
        ia.update(p, info=["main", "biography"])
        dump_item(p, 5)

    print("saving data ...")
    with open(f"all-people-updated.json", "w") as fout:
        json.dump(all_people, fout, indent=2)


if __name__ == "__main__":

    #main()
    get_imdb_people()




