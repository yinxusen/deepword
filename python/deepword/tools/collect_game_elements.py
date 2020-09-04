import random
import re

import fire
import gym
import textworld.gym
from textworld import EnvInfos
from tqdm import tqdm
import numpy as np

from deepword.agents.competition_agent import CompetitionAgent
from deepword.floor_plan import FloorPlanCollector
from deepword.utils import load_game_files
from deepword.agents.utils import INFO_KEY


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
    room_regex = r".*-= (.*) =-.*"
    room_search = re.search(room_regex, master)
    if room_search is not None:
        curr_room = room_search.group(1)
    else:
        curr_room = None
    return curr_room


class CollectorAgent(object):
    @classmethod
    def request_infos(cls):
        raise NotImplementedError()

    def pre_run(self):
        raise NotImplementedError()

    def act(self, obs, scores, dones, infos):
        raise NotImplementedError()

    def post_run(self):
        raise NotImplementedError()


class FPCollector(CollectorAgent):
    def __init__(self):
        self.fp_collector = FloorPlanCollector()
        self.prev_room = None
        self.action = None

    @classmethod
    def request_infos(cls):
        request_infos = EnvInfos(admissible_commands=True)
        return request_infos

    def pre_run(self):
        pass

    def act(self, obs, scores, dones, infos):
        curr_room = get_room(obs[0])
        if self.prev_room is None:
            self.prev_room = curr_room
        if curr_room != self.prev_room:
            self.fp_collector.extend([(self.prev_room, self.action, curr_room)])
            self.prev_room = curr_room
        admissible_actions = infos[INFO_KEY.actions][0]
        go_actions = list(
            filter(
                lambda a: a.startswith("go") or a.startswith("open"),
                admissible_actions))
        # we cannot guarantee go-actions and open-actions
        # can walk all rooms
        if len(go_actions) == 0 or random.random() <= 0.5:
            self.action = random.choice(admissible_actions)
        else:
            self.action = random.choice(go_actions)
        return self.action

    def post_run(self):
        pass


class OneStepCollector(CollectorAgent):
    def __init__(self):
        self.templates = set()
        self.ingredients = set()
        self.total_score = 0
        self.all_max_scores = []

    @classmethod
    def request_infos(cls):
        request_infos = EnvInfos(
            admissible_commands=True,
            command_templates=True,
            max_score=True,
            extras=["recipe"])
        return request_infos

    def pre_run(self):
        pass

    def act(self, obs, scores, dones, infos):
        self.templates.update(infos[INFO_KEY.templates][0])
        self.total_score += infos[INFO_KEY.max_score][0]
        self.all_max_scores.append(infos[INFO_KEY.max_score][0])
        # self.ingredients.update(
        #     CompetitionAgent.get_theme_words(infos[INFO_KEY.recipe][0]))
        action = random.choice(infos[INFO_KEY.actions][0])
        return action

    def post_run(self):
        pass


def run_games(agent, game_files, nb_episodes, max_steps):
    for i in tqdm(range(len(game_files))):
        fg = game_files[i]
        env_id = textworld.gym.register_games(
            [fg], agent.request_infos(),
            batch_size=1,
            max_episode_steps=max_steps,
            name="floor-plan-collector")
        env = gym.make(env_id)
        for j in tqdm(range(nb_episodes)):
            obs, infos = env.reset()
            dones = [False] * len(obs)
            scores = [0] * len(obs)
            while not all(dones):
                action = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = env.step([action])
            agent.act(obs, scores, dones, infos)
        env.close()


class Main(object):
    @classmethod
    def collect_floor_plans(
            cls, save_to, game_dir, f_games=None, nb_episodes=10):
        """
        Collect floor plans with games in game_dir or specified by f_games.

        :param save_to: save to path, must be a npz file, e.g. floor_plan.npz
        :param game_dir:
        :param f_games:
        :param nb_episodes:
        :return:
        """
        agent = FPCollector()
        game_files = load_game_files(game_dir, f_games)
        run_games(agent, game_files, nb_episodes, max_steps=200)
        agent.fp_collector.save_fps(save_to)

    @classmethod
    def collect_others(cls, game_dir, f_games=None, nb_episodes=1):
        agent = OneStepCollector()
        game_files = load_game_files(game_dir, f_games)
        run_games(agent, game_files, nb_episodes, max_steps=1)
        print("ingredients:\n", agent.ingredients)
        print("templates:\n", agent.templates)
        print("total scores: ", agent.total_score)
        print("mean score: {}, std: {}".format(
            np.mean(agent.all_max_scores), np.std(agent.all_max_scores)))


if __name__ == '__main__':
    fire.Fire(Main)
