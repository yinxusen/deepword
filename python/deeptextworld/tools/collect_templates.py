import glob
import os
import sys
import re

import gym
import textworld.gym
from textworld import EnvInfos


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

    for game in game_files:
        obs, infos = env.reset()
        dones = [False] * len(obs)
        master_wo_logo = remove_logo(obs[0]).replace("\n", " ")
        print(master_wo_logo)
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


if __name__ == '__main__':
    game_dir = sys.argv[1]
    game_files = glob.glob(os.path.join(game_dir, "*.ulx"))
    main(game_files)
