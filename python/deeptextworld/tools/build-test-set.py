import os
import sys
import glob
import random


def build_test_set(all_game_dir, all_types):
    for t in all_types:
        files = glob.glob("{}/*-{}-*.ulx".format(all_game_dir, t))
        files = sorted(files)
        random.Random(42).shuffle(files)
        selected = files[-2:]
        print("\n".join(map(
            lambda x: os.path.splitext(os.path.basename(x))[0], selected)))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: argv[1]=f_all_types, argv[2]=all_game_dir")
        exit(1)
    with open(sys.argv[1], "r") as f:
        all_types = map(lambda x: x.strip(), f.readlines())
    build_test_set(sys.argv[2], all_types)