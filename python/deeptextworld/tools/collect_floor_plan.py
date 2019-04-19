import glob
import os
import sys
import re
import random

from tqdm import tqdm
import gym
import textworld.gym
from textworld import EnvInfos

from deeptextworld.floor_plan import FloorPlanCollector
from deeptextworld.train_drrn import load_game_files


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


def get_room(master):
    room_regex = ".*-= (.*) =-.*"
    room_search = re.search(room_regex, master)
    if room_search is not None:
        curr_room = room_search.group(1)
    else:
        curr_room = None
    return curr_room


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

    fp_collector = FloorPlanCollector()

    for i in tqdm(range(len(game_files))):
        fg = game_files[i]
        # print("process file: {}".format(os.path.basename(fg)))
        env_id = textworld.gym.register_games(
            [fg], request_infos,
            max_episode_steps=100,
            name="floor-plan-collector")
        env_id = textworld.gym.make_batch(
            env_id, batch_size=1, parallel=False)
        env = gym.make(env_id)

        fp_collector.add_new_episode(os.path.basename(fg))

        obs, infos = env.reset()
        dones = [False] * len(obs)
        prev_room = None
        action = None
        while not all(dones):
            curr_room = get_room(obs[0])
            if prev_room is None:
                prev_room = curr_room
            if curr_room != prev_room:
                fp_collector.extend([(prev_room, action, curr_room)])
                prev_room = curr_room
            admissible_actions = infos["admissible_commands"][0]
            go_actions = list(filter(lambda a: a.startswith("go"), admissible_actions))
            if len(go_actions) == 0:
                # print("no go action: {}".format(fg))
                action = random.choice(admissible_actions)
            else:
                action = random.choice(go_actions)
            obs, scores, dones, infos = env.step([action])

        if action is not None:
            obs, scores, dones, infos = env.step([action])
    return fp_collector


if __name__ == '__main__':
    game_dir = sys.argv[1]
    f_games = sys.argv[2]
    game_files = load_game_files(game_dir, f_games)
    fp_collector = main(game_files)
    fp_collector.save_fps("/tmp/floor-plan-{}.npz".format(os.path.basename(f_games)))

