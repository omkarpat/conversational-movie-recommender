import re

class DBPedia(object):
    # These correspond to labels that have been misrecognized
    BLACKLIST_URIS = {"http://dbpedia.org/resource/Glossary_of_tennis_terms", 
    "http://dbpedia.org/resource/Good_Movie",
    "http://dbpedia.org/resource/Sierra_Entertainment",
    "http://dbpedia.org/resource/Nice",
    "http://dbpedia.org/resource/Take_Care_(album)",
    "http://dbpedia.org/resource/Cloning",
    "http://dbpedia.org/resource/Blood",
    "http://dbpedia.org/resource/Downhill_creep",
    "http://dbpedia.org/resource/Movies",
    "http://dbpedia.org/resource/Hey_There",
    "http://dbpedia.org/resource/Swimming_(sport)",
    "http://dbpedia.org/resource/Princess_Falls",
    "http://dbpedia.org/resource/Haha_(entertainer)",
    "http://dbpedia.org/resource/LOL",
    "http://dbpedia.org/resource/Drag_queen",
    "http://dbpedia.org/resource/Yea_Football_Club",
    "http://dbpedia.org/resource/Oh_Yeah_(Yello_song)",
    "http://dbpedia.org/resource/Scalable_Coherent_Interface",
    "http://dbpedia.org/resource/CAN_bus",
    "http://dbpedia.org/resource/The_New_One_(horse)",
    "http://dbpedia.org/resource/Information_technology",
    "http://dbpedia.org/resource/The_Glad_Products_Company",
    "http://dbpedia.org/resource/AM_broadcasting",
    "http://dbpedia.org/resource/To_Heart",
    "http://dbpedia.org/resource/National_Organization_for_Women",
    "http://dbpedia.org/resource/Hit_or_Miss_(New_Found_Glory_song)",
    "http://dbpedia.org/resource/Canada",
    "http://dbpedia.org/resource/Different_Things",
    "http://dbpedia.org/resource/Norwegian_Trekking_Association",
    "http://dbpedia.org/resource/Take_One_(Canadian_magazine)",
    "http://dbpedia.org/resource/For_Inspiration_and_Recognition_of_Science_and_Technology",
    "http://dbpedia.org/resource/Two_Guys",
    "http://dbpedia.org/resource/The_Sydney_Morning_Herald",
    "http://dbpedia.org/resource/Booting",
    "http://dbpedia.org/resource/Precious_Time_(album)",
    "http://dbpedia.org/resource/I\\u0027m_Glad",
    "http://dbpedia.org/resource/Social_Democratic_Party_of_Switzerland",
    "http://dbpedia.org/resource/International_Maritime_Organization",
    "http://dbpedia.org/resource/LOL",
    "http://dbpedia.org/resource/Names_of_God_in_Judaism",
    "http://dbpedia.org/resource/Ike_Turner",
    "http://dbpedia.org/resource/Tricky_Stewart",
    "http://dbpedia.org/resource/Movies!",
    }

multi_spaces_pattern = re.compile(r"\s+")

def process_text(text):
    return multi_spaces_pattern.sub(" ", text.capitalize())