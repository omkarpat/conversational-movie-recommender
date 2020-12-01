"""
 Created by diesel
 11/11/20
"""


def main():
    with open("genres.txt", "r") as fin:
        lines = [l for l in fin.read().split("\n") if l]

    all_terms = []
    for line in lines:
        phrases = line.split(";")
        all_terms.extend([p.strip() for p in phrases if p.strip()])

    all_terms = list(set(all_terms))

    with open("genres-clean.txt", "w") as fout:
        fout.write("\n".join(all_terms))

if __name__ == "__main__":
    main()
