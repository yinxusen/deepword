import ast
import os
import re

import fire
from textworld import EnvInfos
from termcolor import colored

from deepword.tools.collect_game_elements import CollectorAgent, run_games
from deepword.agents.utils import INFO_KEY
from deepword.utils import eprint


class ReplayAgent(CollectorAgent):
    def __init__(self, fn_game, fn_log):
        self.eval_results = self.extract_eval_results(fn_log)
        # eprint(self.eval_results[0].keys())
        self.in_game_t = 0
        self.episode_has_started = False
        self.fn_game = fn_game
        self.curr_episode = 0

    @classmethod
    def extract_eval_results(cls, fn_log):
        with open(fn_log, "r") as f:
            lines = f.readlines()
        eval_result_regex = r"^INFO - eval_results: (.*)$"
        matched = []
        for l in lines:
            eval_result_search = re.search(eval_result_regex, l)
            if eval_result_search is not None:
                room_name = eval_result_search.group(1)
                matched.append(
                    ast.literal_eval(
                        room_name.replace("false", "False").replace(
                            "true", "True")))
        return matched

    @classmethod
    def request_infos(cls):
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.command_templates = True
        request_infos.max_score = True
        request_infos.won = True
        request_infos.lost = True
        request_infos.extras = ["recipe"]
        request_infos.admissible_commands = True
        return request_infos

    def pre_run(self):
        pass

    def post_run(self):
        pass

    def start_episode(self):
        self.in_game_t = 0
        self.episode_has_started = True

    def end_episode(self):
        self.episode_has_started = False
        self.curr_episode += 1

    def act(self, obs, scores, dones, infos):
        if not self.episode_has_started:
            self.start_episode()

        if all(dones):
            self.end_episode()
            eprint(colored(obs[0], "cyan"))
            eprint("won: {}".format(infos[INFO_KEY.won][0]))
            eprint("lost: {}".format(infos[INFO_KEY.lost][0]))
            eprint("final score: {}".format(scores[0]))
            eprint("total steps: {}".format(self.in_game_t))
            return

        eprint(colored(obs[0], 'cyan'))
        eprint("score: {}, step: {}".format(scores[0], self.in_game_t))
        eprint()
        # eprint("\t"+"\n\t".join(infos[INFO_KEY.actions][0]))
        eprint()

        action = (
            self.eval_results[0][self.fn_game][self.curr_episode % 2][-1]
            [self.in_game_t])
        eprint(colored("> " + action, "red"))
        self.in_game_t += 1
        return action


def play(game_file, log_file, nb_episodes=2, max_steps=100):
    agent = ReplayAgent(os.path.basename(game_file), log_file)
    run_games(agent, [game_file], nb_episodes, max_steps)


if __name__ == '__main__':
    fire.Fire(play)
