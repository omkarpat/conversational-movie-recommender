import json
import csv

def load_conversations(conversations_file):
    with open(conversations_file, 'r') as conversations_file:
        conversations = []

        for line in conversations_file:
            conversations.append(json.loads(line.strip()))
    
        return conversations

def dump_conversations_to_file(conversations, output_file):
    with open(output_file, 'w') as conversations_file:
        for conversation in conversations:
            conversations_file.write(json.dumps(conversation))
            conversations_file.write('\n')

def load_imdb_list(file_path):
    with open(file_path, 'r') as imdb_file:
        entities = [row["name"] for row in csv.DictReader(imdb_file)]

        return entities

def popular_actors_list():
    actors_list = load_imdb_list('top_1000_actors.csv')

    actors_list.extend([
        "Amy Schumer",
        "David Bowie", # Tesla in The Prestige
        "Chevy Chase", # 
        "Christina Hendricks", # Joan from Mad Men
        "David Tennant", # Dr. Who
        "John Boyega", # Finn from Force Awakens
        "Daisy Ridley", # Rey from Force Awakens
        "Kate McKinnon", # SNL
        "Zach Braff", # Dude from Scrubs
        "Samuel L. Jackson", # Surprised this bad mofo isn't on the list
        "Lizzy Caplan", # Masters of Sex
        "Bela Lugosi", # The iconic dracula
        "Rowan Atkinson", # Mr. Bean, Johnny English
        "Ansel Elgort", # Baby Driver
        "Zac Efron", # High School Musical
        "Kumail Nanjiani", # Dinesh from Silicon Valley
        "Brittany Murphy", # 8 Mile
        "Hank Azaria", # The Simpsons
        "Dolly Parton", #The Best Little Whorehouse in Texas
        "Donald Glover", # Troy from Community
        "Anupam Kher", 
        "Steven Seagal", # Whole bunch of martial arts movies
        "Patrick Dempsey", # Grey's Anatomy
        "Jack Nance", # Eraserhead and Twin Peaks
        "Martin Freeman", # Watson from Sherlock ; Shaun of the Dead
        "Rick Moranis", # Honey I shrunk .. ; Ghostbusters
        "Tiffany Haddish", # Girls Trip
        "Ally Sheedy", # The Breakfast Club 
        "Lin-Manuel Miranda", # Hamilton
        "Ian McShane", # John Wick, Deadpool
        "Elle Fanning", # Maleficent
        "Robert Englund", # Freddie Krueger from a nightmare on Elm Street
        "Tony Jaa", # Ong Bak
        "Shemar Moore", # Criminal Minds
    ])
    return actors_list

def popular_directors_list():
    directors_list =  load_imdb_list('top_250_directors.csv')

    directors_list.extend([
        "Wes Anderson", # Grand Budapest Hotel, Isle of Dogs etc.
        "Wes Craven", # A nightmare on Elm Street
        "George Lucas", # Star wars
        "Edgar Wright", # Baby Driver
        "Tim Burton", # Any movie with Johnny Depp / Helena Bonham Carter
        "Satoshi Kon", # Paprika
        "Trey Parker", "Matt Stone", # South Park, Orgazmo etc.
        "Joss Whedon", # Iron Man, Avengers
        "Guillermo del Toro", # Pacific Rim, The Shape of Water
        "Spike Lee", # BlacKKKlansman, Da Five Bloods
        "John Carpenter", # Escape from New York/LA
        "Werner Herzog", # Nosferatu
        "JJ Abrams", # Star Trek/Star Wars the Force Awakens
        "Guy Ritchie", # Sherlock Holmes
        "Chris Columbus", # Harry Potter
        "Ang Lee", # Crouching Tiger, Hidden Dragon; Brokeback Mountain
        "Jordan Peele", # Get Out
        "Mike Flanagan", # Doctor Sleep, The Haunting of Hill House, Hush
        "Rian Johnson", # Star Wars the Last Jedi
        "Darren Aronofsky", # Black Swan, Requiem for a Dream
        "Zack Snyder", # Man of Steel, Watchmen, Justice League
        "Oliver Stone", # Platoon, Snowden
    ])

    return directors_list