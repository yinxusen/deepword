import glob
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import ChainMap
from collections import namedtuple
from multiprocessing import Pool
from os.path import join as pjoin
from threading import Lock
from typing import List, Dict, Optional, Tuple

import gym
import numpy as np
import textworld.gym
from tensorflow.contrib.training import HParams
from termcolor import colored
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from deeptextworld.agents.utils import INFO_KEY
from deeptextworld.log import Logging
from deeptextworld.stats import mean_confidence_interval
from deeptextworld.utils import agent_name2clazz
from deeptextworld.utils import eprint, report_status


class EvalResult(namedtuple(
        "EvalResult",
        ("score", "positive_score", "negative_score", "max_score",
         "steps", "won", "action_list"))):
    pass


def eval_agent(
        hp: HParams, model_dir: str, load_best: bool, restore_from: str,
        game_files: List[str], gpu_device: Optional[str] = None
) -> Tuple[Dict[str, List[EvalResult]], int]:
    """
    Evaluate an agent with given games.
    For each game, we run nb_episodes, and max_episode_steps for on episode.

    Notice that evaluation game running is different with training.
    In training, we register all given games to TextWorld structure, and play
    them in a random way.
    For evaluation, we register one game at a time, and play it for nb_episodes.

    :param hp: hyperparameter to create the agent
    :param model_dir: model dir of the agent
    :param load_best: bool, load from best_weights or not (last_weights)
    :param restore_from: string, load from a specific model,
    e.g. {model_dir}/last_weights/after_epoch-0
    :param game_files: game files for evaluation
    :param gpu_device: which GPU device to load, in a format of "/device:GPU:i"
    :return: eval_results, loaded_ckpt_step
    """
    eval_results = dict()
    agent_clazz = agent_name2clazz(hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    if gpu_device is not None:
        agent.core.set_d4eval(gpu_device)
    if load_best:  # load from best_weights for evaluation
        agent.eval(load_best=load_best)
    else:  # load from last_weights for dev test
        agent.reset(restore_from)

    requested_infos = agent.select_additional_infos()
    for game_no in range(len(game_files)):
        game_file = game_files[game_no]
        game_name = os.path.basename(game_file)
        env_id = textworld.gym.register_games(
            [game_file], requested_infos, batch_size=1,
            max_episode_steps=hp.game_episode_terminal_t,
            name="eval")
        game_env = gym.make(env_id)
        eprint("eval game: {}".format(game_name))
        assert hp.eval_episode > 0, "no enough episode to eval"
        for episode_no in range(hp.eval_episode):
            action_list = []
            obs, infos = game_env.reset()
            scores = [0] * len(obs)
            dones = [False] * len(obs)
            steps = [0] * len(obs)
            # TODO: make sure verbose won't affect games other than Zork
            # tmp_obs, _, _, _ = game_env.step(["verbose"] * len(obs))
            # eprint("use verbose: {}".format(tmp_obs[0]))
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
            eval_results[game_name].append(EvalResult(
                score=scores[0],
                positive_score=agent.positive_scores,
                negative_score=agent.negative_scores,
                max_score=infos[INFO_KEY.max_score][0],
                steps=steps[0],
                won=infos[INFO_KEY.won][0],
                action_list=action_list))
        game_env.close()
    return eval_results, agent.core.loaded_ckpt_step


def agent_collect_data(
        agent, game_files, max_episode_steps, epoch_size, epoch_limit):
    requested_infos = agent.select_additional_infos()
    env_id = textworld.gym.register_games(
        game_files, requested_infos, batch_size=1,
        max_episode_steps=max_episode_steps,
        name="eval")
    game_env = gym.make(env_id)
    agent.eps = random.random()
    eprint("new randomness: {}".format(agent.eps))

    obs, infos = game_env.reset()
    scores = [0] * len(obs)
    dones = [False] * len(obs)
    for epoch_t in range(epoch_limit):
        for _ in range(epoch_size):
            if not all(dones):
                commands = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = game_env.step(commands)
            else:
                agent.act(obs, scores, dones, infos)
                obs, infos = game_env.reset()
                scores = [0] * len(obs)
                dones = [False] * len(obs)
                agent.eps = random.random()
                eprint("new randomness: {}".format(agent.eps))
        agent.save_snapshot()
        eprint("save snapshot epoch: {}".format(epoch_t))
    game_env.close()


def agg_eval_results(
        eval_results: Dict[str, List[EvalResult]],
        max_steps_per_episode: int = 100
) -> Tuple[Dict[str, EvalResult], float, float, float, float, float, float]:
    """
    Aggregate evaluation results.
    We run N test games, each with M episodes, each episode has a maximum of
    K steps.

    :param eval_results: evaluation results of text-based games, in the
    following format:
      dict(game_name, [eval_result1, evaluate_result2, ..., evaluate_resultM])
      and the number of eval_results are the same for all games.
      evaluate_result:
        score, positive_score, negative_score, max_score, steps, won (bool),
        used_action_list
    :param max_steps_per_episode: i.e. M, default = 100

    :return:
      agg_per_game:
        dict(game_name, sum scores, sum max scores, sum steps, # won)
      sample_mean: total earned scores / total maximum scores
      confidence_interval: confidence interval of sample_mean over M episodes.
      steps: total used steps / total maximum steps
    """
    agg_per_game = {}
    total_scores_per_episode = None  # np array of shape M
    total_positive_scores = 0
    total_negative_scores = 0
    total_steps = 0
    max_scores_per_episode = 0
    total_episodes = 0
    total_won = 0
    for game_name in eval_results:
        res = eval_results[game_name]
        agg_score = np.asarray(list(map(lambda r: r.score, res)))
        agg_positive_score = sum(map(lambda r: r.positive_score, res))
        agg_negative_score = sum(map(lambda r: r.negative_score, res))
        # all max scores should be equal, so just pick anyone
        agg_max_score = max(map(lambda r: r.max_score, res))
        max_scores_per_episode += agg_max_score
        n_episodes = len(res)
        total_episodes += n_episodes
        agg_step = sum(map(lambda r: r.steps, res))
        agg_nb_won = len(list(filter(lambda r: r.won, res)))
        total_won += agg_nb_won
        agg_per_game[game_name] = EvalResult(
            score=np.sum(agg_score),
            positive_score=agg_positive_score,
            negative_score=agg_negative_score,
            max_score=agg_max_score * n_episodes,
            steps=agg_step,
            won=agg_nb_won,
            action_list=None)
        if total_scores_per_episode is None:
            total_scores_per_episode = np.zeros_like(agg_score)
        total_scores_per_episode += agg_score
        total_positive_scores += agg_positive_score
        total_negative_scores += agg_negative_score
        total_steps += agg_step
    max_steps = total_episodes * max_steps_per_episode
    total_scores_percentage, confidence_interval = mean_confidence_interval(
        total_scores_per_episode / max_scores_per_episode)
    return (agg_per_game, total_scores_percentage, confidence_interval,
            total_positive_scores, total_negative_scores,
            total_steps * 1. / max_steps, total_won * 1. / total_episodes)


def scores_of_tiers(agg_per_game: Dict[str, EvalResult]) -> Dict[str, float]:
    """
    Compute scores per tier given aggregated scores per game
    :param agg_per_game:
    :return: list of tier-name -> scores, starting from tier1 to tier6
    """
    games = agg_per_game.keys()

    tiers2games = {
        "tier1": list(
            filter(lambda k: "go" not in k and "recipe1" in k, games)),
        "tier2": list(
            filter(lambda k: "go" not in k and "recipe2" in k, games)),
        "tier3": list(
            filter(lambda k: "go" not in k and "recipe3" in k, games)),
        "tier4": list(filter(lambda k: "go6" in k, games)),
        "tier5": list(filter(lambda k: "go9" in k, games)),
        "tier6": list(filter(lambda k: "go12" in k, games))
    }

    tiers2scores = dict()

    for k_tier in tiers2games:
        if not tiers2games[k_tier]:
            continue
        earned = 0
        total = 0
        for g in tiers2games[k_tier]:
            earned += agg_per_game[g].score
            total += agg_per_game[g].max_score
        tiers2scores[k_tier] = earned * 1. / total
    return tiers2scores


class MultiGPUsEvalPlayer(Logging):
    """
    Eval Player that runs on multiple GPUs
    """
    def __init__(self, hp, model_dir, game_files, n_gpus, load_best=True):
        super(MultiGPUsEvalPlayer, self).__init__()
        self.hp = hp
        self.prev_best_scores = 0
        self.prev_best_steps = sys.maxsize
        self.model_dir = model_dir
        self.gpu_devices = ["/device:GPU:{}".format(i) for i in range(n_gpus)]
        self.game_files = game_files
        self.portion_files = self.split_game_files(game_files, n_gpus)
        self.load_best = load_best

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

    def evaluate(self, restore_from, debug=False):
        self.debug("start evaluation ...")
        results = []
        if debug:
            res = eval_agent(
               self.hp, self.model_dir, self.load_best, restore_from,
               self.game_files, self.gpu_devices[0])
            results.append(res)
        else:
            pool = Pool(len(self.portion_files))
            async_results = [
                pool.apply_async(
                    eval_agent,
                    (self.hp, self.model_dir, self.load_best, restore_from,
                     files, gpu_device))
                for files, gpu_device in zip(
                    self.portion_files, self.gpu_devices)]
            for res in async_results:
                try:
                    results.append(res.get())
                except Exception as e:
                    self.error(
                        "evaluation error with {}\n{}".format(restore_from, e))
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(
                        exc_type, exc_value, exc_traceback, limit=None,
                        file=sys.stdout)
                    pool.terminate()
                    self.debug("multi-process pool terminated.")
                    return
            pool.close()
            pool.join()
            self.debug("evaluation pool closed")

        loaded_steps = [res[1] for res in results]
        assert len(set(loaded_steps)) == 1, "load different versions of model"

        results = [res[0] for res in results]
        # TODO: same key will be updated by ChainMap
        eval_results = dict(ChainMap(*results))
        (agg_res, total_scores, confidence_intervals, total_positive_scores,
         total_negative_scores, total_steps, n_won
         ) = agg_eval_results(eval_results)
        self.info("eval_results: {}".format(eval_results))
        self.info("eval aggregated results: {}".format(agg_res))
        self.info(report_status([
            ("after-epoch", loaded_steps[0]),
            ("scores", "{:.2f}".format(total_scores)),
            ("confidence", "{:.2f}".format(confidence_intervals)),
            ("positive scores", "{:.2f}".format(total_positive_scores)),
            ("negative scores", "{:.2f}".format(total_negative_scores)),
            ("steps", "{:.2f}".format(total_steps)),
            ("n_won", "{:.2f}".format(n_won))
        ]))
        tiers2scores = scores_of_tiers(agg_res)
        tiers2scores = sorted(list(tiers2scores.items()), key=lambda x: x[0])
        self.info("scores per tiers:\n{}".format(tiers2scores))

        if not self.load_best:
            if self.has_better_model(total_scores, total_steps):
                self.info(
                    "found better agent, save model after-epoch-{}".format(
                        loaded_steps[0]))
                self.prev_best_scores = total_scores
                self.prev_best_steps = total_steps
                # copy best model so far
                try:
                    self.save_best_model(loaded_steps[0])
                except Exception as e:
                    self.warning("save best model error:\n{}".format(e))
            else:
                self.info("no better model, pass ...")

    def save_best_model(self, loaded_ckpt_step):
        ckpt_path = pjoin(self.model_dir, "last_weights")
        best_path = pjoin(self.model_dir, "best_weights")
        if not os.path.exists(best_path):
            os.mkdir(best_path)
        for file in glob.glob(
                pjoin(ckpt_path, "after-epoch-{}*".format(loaded_ckpt_step))):
            dst = shutil.copy(file, best_path)
            self.debug("copied: {} -> {}".format(file, dst))
        ckpt_file = pjoin(ckpt_path, "checkpoint")
        dst = shutil.copy(ckpt_file, best_path)
        self.debug("copied: {} -> {}".format(ckpt_file, dst))


class NewModelHandler(FileSystemEventHandler):
    def __init__(self, hp, model_dir, game_files, n_gpus):
        self.hp = hp
        self.model_dir = model_dir
        self.game_files = game_files
        self.n_gpus = n_gpus
        self.eval_player = None
        self.lock = Lock()
        self.watched_file = "checkpoint"

    def run_eval_player(self, restore_from=None, load_best=False):
        if self.eval_player is None:
            self.eval_player = MultiGPUsEvalPlayer(
                self.hp, self.model_dir, self.game_files, self.n_gpus,
                load_best=load_best)
        time.sleep(10)  # wait until all files of a model has been saved
        if self.lock.locked():
            eprint("Give up evaluation since model is running.")
            return
        self.lock.acquire()
        try:
            self.eval_player.evaluate(restore_from)
        except Exception as e:
            eprint("evaluation failed with {}\n{}".format(restore_from, e))
        self.lock.release()

    def is_ckpt_file(self, src_path):
        return self.watched_file == os.path.basename(src_path)

    def on_created(self, event):
        eprint("create ", event.src_path)
        if not event.is_directory and self.is_ckpt_file(event.src_path):
            self.run_eval_player()

    def on_modified(self, event):
        eprint("modify", event.src_path)
        if not event.is_directory and self.is_ckpt_file(event.src_path):
            self.run_eval_player()


class WatchDogEvalPlayer(Logging):
    def __init__(self):
        super(WatchDogEvalPlayer, self).__init__()

    def start(self, hp, model_dir, game_files, n_gpus):
        event_handler = NewModelHandler(hp, model_dir, game_files, n_gpus)
        watched_dir = pjoin(model_dir, "last_weights")
        if not os.path.exists(watched_dir):
            os.mkdir(watched_dir)
        self.debug("watch on {}".format(watched_dir))
        observer = Observer()
        observer.schedule(event_handler, watched_dir, recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(10)
                self.debug("watching ...")
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


class LoopDogEvalPlayer(Logging):
    def __init__(self):
        super(LoopDogEvalPlayer, self).__init__()
        self.file_content = hash("")

    def start(self, hp, model_dir, game_files, n_gpus):
        event_handler = NewModelHandler(hp, model_dir, game_files, n_gpus)
        watched_file = pjoin(model_dir, "last_weights", "checkpoint")
        self.debug("watch on {}".format(watched_file))
        try:
            while True:
                time.sleep(10)
                self.debug("watching ...")
                try:
                    with open(watched_file, 'rb') as f:
                        content = hash(f.read())
                    if content != self.file_content:
                        self.debug(
                            "encounter new file {} -> {} for evaluation".format(
                                self.file_content, content))
                        self.file_content = content
                        event_handler.run_eval_player()
                    else:
                        pass
                except Exception as e:
                    self.warning("cannot read watched file: {}\n{}".format(
                        watched_file, e))
        except KeyboardInterrupt:
            pass


class FullDirEvalPlayer(Logging):
    def __init__(self):
        super(FullDirEvalPlayer, self).__init__()

    @classmethod
    def start(
            cls, hp, model_dir, game_files, n_gpus,
            range_min=None, range_max=None):
        watched_files = pjoin(model_dir, "last_weights", "after-epoch-*.index")
        files = [os.path.splitext(fn)[0] for fn in glob.glob(watched_files)]
        if len(files) == 0:
            eprint(colored("No checkpoint found!", "red"))
            return
        step2ckpt = dict(map(lambda fn: (int(fn.split("-")[-1]), fn), files))
        steps = sorted(list(step2ckpt.keys()))
        if range_max is None:
            range_max = steps[-1]
        if range_min is None:
            range_min = steps[0]
        steps = [step for step in steps if range_min <= step <= range_max]
        eprint("valid evaluation steps: {}".format(
            ",".join([str(step) for step in steps])))

        event_handler = NewModelHandler(hp, model_dir, game_files, n_gpus)
        for step in steps:
            event_handler.run_eval_player(step2ckpt[step])
