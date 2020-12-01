"""
 Created by diesel
 12/19/19
"""


from __future__ import print_function, division


import os
import pandas as pd


def check_and_create(out_dir):
    if not os.path.exists(out_dir):
        print(" ** creating directory: {}".format(out_dir))
        os.makedirs(out_dir)


def write_lines(outpath, lines):
    with open(outpath, "w") as fout:
        fout.write("\n".join(lines))


def read_lines(fpath):
    with open(fpath, "r") as fin:
        lines = fin.read().split("\n")
    return lines


def read_csv_shards(dirpath, name_prefix, postfix):
    #print("reading csv shards")
    #print(" * dirpath:", dirpath)
    #print(" * name_prefix:", name_prefix)
    #print(" * postfix:", postfix)

    filenames = [fname for fname in os.listdir(dirpath) if fname.startswith(name_prefix) if fname.endswith(postfix)]
    #print(" * filenames:", filenames)

    idxs = [fname[len(name_prefix):-len(postfix)] for fname in filenames]
    #print(" * idxs:", idxs)
    sorted_names = [(int(idx), name) for idx, name in zip(idxs, filenames)]
    sorted_names.sort()

    data = pd.DataFrame()
    for idx, fname in sorted_names:
        df = pd.read_csv(os.path.join(dirpath, fname))
        data = pd.concat([data, df], axis=0)

    data.reset_index(inplace=True, drop=True)
    return data


def save_df_shards(to_write, outpath, max_size=15000):

    h = 0

    while to_write:
        h += 1
        new_df = pd.DataFrame(to_write[:max_size])
        outfile = outpath + "-{}.csv".format(h)
        if "Index" in new_df.columns.values:
            new_df.drop("Index", axis=1, inplace=True)
        new_df.to_csv(outfile, index=False, encoding="utf-8")
        to_write = to_write[max_size:]


def main():
    pass


if __name__ == "__main__":
    main()
