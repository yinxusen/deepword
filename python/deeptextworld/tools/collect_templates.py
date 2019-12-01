import glob
import os
import sys
import re
import random

import gym
import textworld.gym
from textworld import EnvInfos
from deeptextworld.agents.base_agent import BaseAgent


def contain_words(sentence, words):
    return any(map(lambda w: w in sentence, words))

def contain_theme_words(theme_words, actions):
    if theme_words is None:
        return actions
    contained = []
    others = []
    for a in actions:
        if contain_words(a, theme_words):
            contained.append(a)
        else:
            others.append(a)

    return contained, others


def remove_logo(first_master):
    lines = first_master.split("\n")
    start_line = 0
    room_regex = "^\s*-= (.*) =-.*"
    for i, l in enumerate(lines):
        room_search = re.search(room_regex, l)
        if room_search is not None:
            start_line = i
            break
        else:
            pass
    modified_master = "\n".join(lines[start_line:])
    return modified_master


def main(game_files):
    templates = set()
    request_infos = EnvInfos()
    request_infos.description = True
    request_infos.inventory = True
    request_infos.entities = True
    request_infos.verbs = True
    request_infos.command_templates = True
    request_infos.max_score = True
    request_infos.has_won = True
    request_infos.extras = ["recipe"]
    request_infos.admissible_commands = True

    env_id = textworld.gym.register_games(
        game_files, request_infos,
        max_episode_steps=1,
        name="human")
    env_id = textworld.gym.make_batch(
        env_id, batch_size=1, parallel=False)
    env = gym.make(env_id)
    total_ingredients = set()
    for game in game_files:
        obs, infos = env.reset()
        dones = [False] * len(obs)
        master_wo_logo = remove_logo(obs[0]).replace("\n", " ")
        # print(master_wo_logo)
        total_ingredients.update(BaseAgent.get_theme_words(infos["extra.recipe"][0]))
            # print(ingredient)
        # # print("max score is {}".format(infos["max_score"][0]))
        # theme_regex = ".*Ingredients:<\|>(.*)<\|>Directions.*"
        # theme_words_search = re.search(theme_regex, infos["extra.recipe"][0].replace("\n", "<|>"))
        # if theme_words_search:
        #     theme_words = theme_words_search.group(1)
        #     theme_words = list(
        #         filter(lambda w: w != "",
        #                map(lambda w: w.strip(), theme_words.split("<|>"))))
        # else:
        #     theme_words = None

        # templates.update(infos["command_templates"][0])

    # print("\n".join(sorted(templates)))
    print("\n".join(total_ingredients))


def split_train_dev(game_files):
    """
    Split train/dev sets from given game files
    sort - shuffle w/ Random(42) - 90%/10% split
      - if #game_files < 10, then use the last one as dev set;
      - if #game_files == 1, then use the one as both train and dev.
    :param game_files: game files
    :return: None if game_files is empty, otherwise (train, dev)
    """
    # have to sort first, otherwise after shuffling the result is different
    # on different platforms, e.g. Linux VS MacOS.
    game_files = sorted(game_files)
    random.Random(42).shuffle(game_files)
    if len(game_files) == 0:
        print("no game files found!")
        return None
    elif len(game_files) == 1:
        train_games = game_files
        dev_games = game_files
    elif len(game_files) < 10:  # use the last one as eval
        train_games = game_files[:-1]
        dev_games = game_files[-1:]
    else:
        num_train = int(len(game_files) * 0.9)
        train_games = game_files[:num_train]
        dev_games = game_files[num_train:]
    return train_games, dev_games


def load_game_files(game_dir, f_games=None):
    """
    Choose games appearing in f_games in a given game dir. Return all games in
    the game dir if f_games is None.
    :param game_dir: a dir
    :param f_games: a file of game names
    :return: a list of games
    """
    if f_games is not None:
        with open(f_games, "r") as f:
            selected_games = map(lambda x: x.strip(), f.readlines())
        game_files = list(map(
            lambda x: os.path.join(game_dir, "{}.ulx".format(x)),
            selected_games))
    else:
        game_files = glob.glob(os.path.join(game_dir, "*.ulx"))
    return game_files


if __name__ == '__main__':
    game_dir = sys.argv[1]
    if len(sys.argv) > 2:
        f_games = sys.argv[2]
        game_files = load_game_files(game_dir, f_games)
        games = split_train_dev(game_files)
        if games is None:
            exit(-1)
        train_games, dev_games = games
        game_files = train_games
    else:
        game_files = glob.glob(os.path.join(game_dir, "*.ulx"))
    main(game_files)
