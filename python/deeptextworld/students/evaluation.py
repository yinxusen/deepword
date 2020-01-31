import os
import sys

import gym
import textworld.gym

from deeptextworld.agents import dqn_agent
from deeptextworld.students.utils import agg_results
from deeptextworld.utils import ctime
from deeptextworld.utils import eprint


class EvalPlayer(object):
    """
    Evaluate agent with games.
    """
    def __init__(self, hp, model_dir, game_files):
        self.hp = hp
        self.game_files = game_files
        self.game_names = [os.path.basename(fn) for fn in game_files]
        agent_clazz = getattr(dqn_agent, hp.agent_clazz)
        self.agent = agent_clazz(hp, model_dir)
        self.prev_total_scores = 0
        self.prev_total_steps = sys.maxsize

    def evaluate(self, nb_episodes):
        """
        Run nb_episodes times of all games for evaluation.
        :param nb_episodes:
        :return:
        """
        self.agent.reset()
        eval_start_t = ctime()
        eval_results = eval_agent(
            self.agent, self.game_files, nb_episodes,
            self.hp.game_episode_terminal_t)
        eval_end_t = ctime()
        (agg_res, total_scores, confidence_intervals, total_steps,
         n_won) = agg_results(eval_results)
        eprint("eval_results: {}".format(eval_results))
        eprint("eval aggregated results: {}".format(agg_res))
        eprint(
            "after-epoch: {}, scores: {:.2f}, confidence: {:2f}, steps: {:.2f},"
            " n_won: {:.2f}".format(
                self.agent.loaded_ckpt_step, total_scores, confidence_intervals,
                total_steps, n_won))
        eprint(
            "time to finish eval: {}".format(eval_end_t - eval_start_t))
        if ((total_scores > self.prev_total_scores) or
                ((total_scores == self.prev_total_scores) and
                 (total_steps < self.prev_total_steps))):
            eprint("found better agent, save model ...")
            self.prev_total_scores = total_scores
            self.prev_total_steps = total_steps
            self.agent.save_best_model()
        else:
            eprint("no better model, pass ...")


def eval_agent(agent, game_files, nb_episodes, max_episode_steps):
    """
    Evaluate an agent with given games.
    For each game, we run nb_episodes, and max_episode_steps for on episode.

    Notice that evaluation game running is different with training.
    In training, we register all given games to TextWorld structure, and play
    them in a random way.
    For evaluation, we register one game at a time, and play it for nb_episodes.

    :param agent: A TextWorld Agent
    :param game_files: TextWorld game files
    :param nb_episodes:
    :param max_episode_steps:
    :return: evaluation results. It's a dict with game_names as keys, and the
    list of tuples (earned_scores, max_scores, used_steps, has_won, action_list)
    """
    eval_results = dict()
    requested_infos = agent.select_additional_infos()
    for game_no in range(len(game_files)):
        game_file = game_files[game_no]
        game_name = os.path.basename(game_file)
        env_id = textworld.gym.register_games(
            [game_file], requested_infos,
            max_episode_steps=max_episode_steps,
            name="eval")
        env_id = textworld.gym.make_batch(env_id, batch_size=1, parallel=False)
        game_env = gym.make(env_id)
        eprint("eval game: {}".format(game_name))

        for episode_no in range(nb_episodes):
            action_list = []
            obs, infos = game_env.reset()
            scores = [0] * len(obs)
            dones = [False] * len(obs)
            steps = [0] * len(obs)
            while not all(dones):
                # Increase step counts.
                steps = ([step + int(not done)
                          for step, done in zip(steps, dones)])
                commands = agent.act(obs, scores, dones, infos)
                action_list.append(commands[0])
                obs, scores, dones, infos = game_env.step(commands)

            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)

            if game_name not in eval_results:
                eval_results[game_name] = []
            eval_results[game_name].append(
                (scores[0], infos["max_score"][0], steps[0],
                 infos["has_won"][0], action_list))
    return eval_results


