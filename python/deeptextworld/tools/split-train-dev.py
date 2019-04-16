import glob
import os
import sys

from deeptextworld.train_drrn import split_train_dev


def split(f_games):
    with open(f_games, "r") as f:
        fnames = map(lambda n: n.strip(), f.readlines())

    games = split_train_dev(fnames)
    if games is not None:
        train_set, dev_set = games
        with open("{}-train".format(f_games), "w") as ft:
            ft.write("\n".join(sorted(train_set)) + "\n")
        with open("{}-dev".format(f_games), "w") as fd:
            fd.write("\n".join(sorted(dev_set)) + "\n")


if __name__ == "__main__":
    files_of_games = glob.glob(os.path.join(sys.argv[1], "*.games.txt"))
    for f in files_of_games:
        split(f)
