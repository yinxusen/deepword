import glob
import os
import random

import fire


def build_test_set(f_all_types, all_game_dir):
    """
    Random select two games for each type as test set
    :param f_all_types: file contains type info, one per line
    :param all_game_dir: dir contains all games
    :return: names of selected test games, without file extension
    """
    with open(f_all_types, "r") as f:
        all_types = map(lambda x: x.strip(), f.readlines())
    for t in all_types:
        files = glob.glob("{}/*-{}-*.ulx".format(all_game_dir, t))
        files = sorted(files)
        random.Random(42).shuffle(files)
        selected = files[-2:]
        print("\n".join(map(
            lambda x: os.path.splitext(os.path.basename(x))[0], selected)))


if __name__ == '__main__':
    fire.Fire(build_test_set)
