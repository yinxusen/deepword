import glob
import os
import sys


def diff_names(f_games, test_set):
    with open(f_games, "r") as f:
        fnames = map(lambda n: n.strip(), f.readlines())
    with open(test_set, "r") as f:
        test_names = set(map(lambda n: n.strip(), f.readlines()))

    diff = filter(lambda n: n not in test_names, fnames)
    with open("{}-diff".format(f_games), "w") as ft:
        ft.write("\n".join(sorted(diff)) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: argv[1]=game_files_dir, sys.argv[2]=test_set")
        exit(1)
    files_of_games = glob.glob(os.path.join(sys.argv[1], "*.games.txt"))
    for f in files_of_games:
        diff_names(f, sys.argv[2])
