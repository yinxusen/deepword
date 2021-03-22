from os import path

import fire

dir_path = path.dirname(path.realpath(__file__))
project_path = path.join(dir_path, "../../..")
fn_game_ids = path.join(
    project_path, "resources/miscellany/cooking_train_game_id_n_starter.txt")


def game_name2id(f_games):
    with open(f_games, 'r') as f:
        game_names = [x.strip() for x in f.readlines()]

    with open(fn_game_ids) as f:
        game_ids = [x.strip() for x in f.readlines()]
        game_ids = [x.split("\t") for x in game_ids]
        game_ids = dict([(x[0], x[1]) for x in game_ids])

    for gn in game_names:
        print("{}\t{}".format(gn, game_ids[gn]))


if __name__ == '__main__':
    fire.Fire(game_name2id)
