import math
import os
import random
import sys
import time
from collections import ChainMap
from multiprocessing import Pool
from os.path import join as pjoin
from threading import Lock
from threading import Thread

import gym
import textworld.gym
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from deeptextworld.agents import dqn_agent
from deeptextworld.hparams import load_hparams_for_evaluation
from deeptextworld.students.utils import agg_results
from deeptextworld.utils import ctime
from deeptextworld.utils import eprint


class EvalPlayer(object):
    """
    Evaluate agent with games.
    """
    def __init__(self, hp, model_dir, game_files, gpu_device=None):
        self.hp = hp
        self.game_files = game_files
        self.model_dir = model_dir
        self.game_names = [os.path.basename(fn) for fn in game_files]
        agent_clazz = getattr(dqn_agent, hp.agent_clazz)
        self.agent = agent_clazz(hp, model_dir)
        if gpu_device is not None:
            self.agent.set_d4eval(gpu_device)

    def reload(self):
        """
        Reload model for agent

        :return: loaded checkpoint step
        """
        self.agent.reset()
        return self.agent.loaded_ckpt_step

    def evaluate(self, results):
        """
        Run nb_episodes times of all games for evaluation.
        :return:
        """
        eval_start_t = ctime()
        eval_results = eval_agent(
            self.agent, self.game_files, self.hp.eval_episode,
            self.hp.game_episode_terminal_t)
        results.append(eval_results)
        eval_end_t = ctime()
        eprint(
            "time to finish eval: {}".format(eval_end_t - eval_start_t))
        return eval_results


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


class MultiGPUsEvalPlayer(object):
    def __init__(self, hp, model_dir, game_files, gpu_devices):
        self.prev_best_scores = 0
        self.prev_best_steps = sys.maxsize
        self.evaluated_epochs = dict()
        self.portion_files = self.split_game_files(game_files, len(gpu_devices))
        self.players = [
            EvalPlayer(hp, model_dir, files, gpu_device)
            for gpu_device, files in zip(gpu_devices, self.portion_files)]
        self.pool = Pool(len(self.portion_files))

    @classmethod
    def split_game_files(cls, game_files, k, rnd_seed=42):
        """
        Split game files into k portions for multi GPUs playing
        :param game_files:
        :param k:
        :param rnd_seed:
        :return:
        """
        game_files = sorted(game_files)
        random.Random(rnd_seed).shuffle(game_files)
        n_files = len(game_files)
        if n_files == 0:
            raise ValueError("no game files found!")

        portion = math.ceil(len(game_files) / k)
        files = [
            game_files[i * portion: min(len(game_files), (i + 1) * portion)]
            for i in range(k)]
        return files

    def has_better_model(self, total_scores, total_steps):
        has_better_score = total_scores > self.prev_best_scores
        has_fewer_steps = (
                total_scores == self.prev_best_scores and
                total_steps < self.prev_best_steps)
        return has_better_score or has_fewer_steps

    def evaluate_with_reload(self):
        loaded_steps = [player.reload() for player in self.players]
        assert len(set(loaded_steps)) == 1, "eval players load different models"
        if loaded_steps[0] in self.evaluated_epochs:
            eprint("model {} has been evaluated".format(loaded_steps[0]))
        else:
            results = []
            workers = [Thread(target=player.evaluate, args=(results,))
                       for player in self.players]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            eval_results = dict(ChainMap(*results))
            (agg_res, total_scores, confidence_intervals, total_steps,
             n_won) = agg_results(eval_results)
            eprint("eval_results: {}".format(eval_results))
            eprint("eval aggregated results: {}".format(agg_res))
            eprint(
                "after-epoch: {}, scores: {:.2f}, confidence: {:2f},"
                " steps: {:.2f}, n_won: {:.2f}".format(
                    loaded_steps[0], total_scores, confidence_intervals,
                    total_steps, n_won))

            if self.has_better_model(total_scores, total_steps):
                eprint("found better agent, save model ...")
                self.prev_best_scores = total_scores
                self.prev_best_steps = total_steps
                self.players[0].agent.save_best_model()
            else:
                eprint("no better model, pass ...")


class NewModelHandler(FileSystemEventHandler):
    def __init__(self, cmd_args, model_dir, game_files, gpu_devices):
        self.cmd_args = cmd_args
        self.model_dir = model_dir
        self.game_files = game_files
        self.gpu_devices = gpu_devices
        self.eval_player = None
        self.lock = Lock()
        self.watched_file = pjoin("last_weights", "checkpoint")

    def run_eval_player(self, event):
        if self.eval_player is None:
            config_file = pjoin(self.model_dir, "hparams.json")
            hp = load_hparams_for_evaluation(config_file, self.cmd_args)
            self.eval_player = MultiGPUsEvalPlayer(
                hp, self.model_dir, self.game_files, self.gpu_devices)
        else:
            pass
        time.sleep(10)  # wait until all files of a model has been saved
        if self.lock.locked():
            eprint("Give up evaluation since model {} is running.".format(
                self.eval_player.players[0].agent.loaded_ckpt_step))
            return
        self.lock.acquire()
        eprint("eval caused by modified file: {}".format(event.src_path))
        self.eval_player.evaluate_with_reload()
        self.lock.release()

    def is_ckpt_file(self, src_path):
        return self.watched_file in src_path

    def on_created(self, event):
        eprint("create ", event.src_path)
        if not event.is_directory and self.is_ckpt_file(event.src_path):
            self.run_eval_player(event)

    def on_modified(self, event):
        eprint("modify", event.src_path)
        if not event.is_directory and self.is_ckpt_file(event.src_path):
            self.run_eval_player(event)


class WatchDogEvalPlayer(object):
    def start(self, cmd_args, model_dir, game_files, gpu_devices):
        event_handler = NewModelHandler(
            cmd_args, model_dir, game_files, gpu_devices)
        observer = Observer()
        observer.schedule(event_handler, model_dir, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
