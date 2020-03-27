from __future__ import print_function

import glob
import hashlib
import logging
import logging.config
import logging.handlers
import math
import os
import random
import sys
import time
from itertools import chain
from typing import List, Tuple

import numpy as np
import ruamel.yaml

from deeptextworld.stats import mean_confidence_interval


def get_hash(txt: str) -> str:
    """
    Compute hash value for a text as a label
    :param txt:
    :return:
    """
    return hashlib.md5(txt.encode("utf-8")).hexdigest()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def flatmap(f, items):
    return list(chain.from_iterable(map(f, items)))


def flatten(items):
    return list(chain.from_iterable(items))


def uniq(lst):
    """
    this is an order-preserving unique
    """
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]


def load_uniq_lines(fname):
    with open(fname, 'r') as f:
        lines = map(lambda l: l.strip(), f.readlines())
    return uniq(lines)


def load_vocab(vocab_file):
    return load_uniq_lines(vocab_file)


def load_actions(action_file):
    return load_uniq_lines(action_file)


def get_token2idx(tokens):
    uniq_tokens = uniq(tokens)
    return dict(map(
        lambda idx_token: (idx_token[1], idx_token[0]), enumerate(uniq_tokens)))


def col(memory, idx):
    """
    get column from index of a memory list, or a mini-batch of memory list
    """
    return list(map(lambda m: m[idx], memory))


def ctime():
    """
    current time in millisecond
    """
    return int(round(time.time() * 1000))


def setup_logging(
        default_path='logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG',
        local_log_filename=None):

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = ruamel.yaml.safe_load(f.read())
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    # add more handlers
    if local_log_filename is not None:
        rh = logging.handlers.RotatingFileHandler(
            local_log_filename, maxBytes=100*1024*1024, backupCount=100)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        rh.setFormatter(formatter)
        logging.getLogger().addHandler(rh)
    # suppress log from stanford corenlp
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


def model_name2clazz(name):
    """
    Find the class given the model name in this package.

    :param name:
    :return:
    """
    from deeptextworld.models import dqn_model, drrn_model, dsqn_model, \
        gen_model, commonsense_model, legacy_models

    for namespace in [dqn_model, drrn_model, dsqn_model, gen_model,
                      commonsense_model, legacy_models]:
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise ValueError("{} not found in models".format(name))


def learner_name2clazz(name):
    from deeptextworld.students import student_learner, bert_swag

    for namespace in [student_learner, bert_swag]:
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise ValueError("{} not found in student learners".format(name))


def agent_name2clazz(name):
    """
    Find the class given the model name in this package.

    :param name:
    :return:
    """
    from deeptextworld.agents import dqn_agent, drrn_agent, dsqn_agent, \
        gen_agent, commonsense_agent

    for namespace in [dqn_agent, drrn_agent, dsqn_agent, gen_agent,
                      commonsense_agent]:
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise ValueError("{} not found in agents".format(name))


def core_name2clazz(name):
    return agent_name2clazz(name)


def agg_results(eval_results, max_steps_per_episode=100):
    """
    Aggregate evaluation results.
    We run N test games, each with M episodes, each episode has a maximum of
    K steps.

    :param eval_results: evaluation results of text-based games, in the
    following format:
      dict(game_name, [eval_result1, evaluate_result2, ..., evaluate_resultM])
      and the number of eval_results are the same for all games.
      evaluate_result:
        score, max_score, steps, won (bool), used_action_list
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
    total_steps = 0
    max_scores_per_episode = 0
    total_episodes = 0
    total_won = 0
    for game_name in eval_results:
        res = eval_results[game_name]
        agg_score = np.asarray(list(map(lambda r: r[0], res)))
        # all max scores should be equal, so just pick anyone
        agg_max_score = max(map(lambda r: r[1], res))
        max_scores_per_episode += agg_max_score
        n_episodes = len(res)
        total_episodes += n_episodes
        agg_step = sum(map(lambda r: r[2], res))
        agg_nb_won = len(list(filter(lambda r: r[3], res)))
        total_won += agg_nb_won
        agg_per_game[game_name] = (
            np.sum(agg_score), agg_max_score * n_episodes,
            agg_step, agg_nb_won)
        if total_scores_per_episode is None:
            total_scores_per_episode = np.zeros_like(agg_score)
        total_scores_per_episode += agg_score
        total_steps += agg_step
    max_steps = total_episodes * max_steps_per_episode
    sample_mean, confidence_interval = mean_confidence_interval(
        total_scores_per_episode / max_scores_per_episode)
    return (agg_per_game, sample_mean, confidence_interval,
            total_steps * 1. / max_steps, total_won * 1. / total_episodes)


def scores_of_tiers(agg_per_game):
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
            earned += agg_per_game[g][0]
            total += agg_per_game[g][1]
        tiers2scores[k_tier] = earned * 1. / total
    tiers2scores = sorted(list(tiers2scores.items()), key=lambda x: x[0])
    return tiers2scores


def split_train_dev(game_files, train_ratio=0.9, rnd_seed=42):
    """
    Split train/dev sets from given game files
    sort - shuffle w/ Random(42) - split

    :param game_files: game files
    :param train_ratio: the percentage of training files.
    :param rnd_seed: for randomly shuffle files, default = 42
    :return: train_games, dev_games
    :exception: empty game_files
    """
    # have to sort first, otherwise after shuffling the result is different
    # on different platforms, e.g. Linux VS MacOS.
    game_files = sorted(game_files)
    random.Random(rnd_seed).shuffle(game_files)
    n_files = len(game_files)
    if n_files == 0:
        raise ValueError("no game files found!")

    n_train = int(math.ceil(n_files * train_ratio))
    n_dev = n_files * (1 - train_ratio)
    n_dev = int(math.floor(n_dev)) if n_dev > 1 else 1
    train_games = game_files[:n_train]
    dev_games = game_files[-n_dev:]
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


def load_and_split(game_path: str, f_games: str) -> Tuple[List[str], List[str]]:
    """
    Load games and split train dev set
    :param game_path:
    :param f_games:
    :return:
    """
    game_files = load_game_files(game_path, f_games)
    train_games, dev_games = split_train_dev(game_files)
    return train_games, dev_games


def setup_train_log(model_dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_config_file = '{}/../../../conf/logging.yaml'.format(current_dir)
    setup_logging(
        default_path=log_config_file,
        local_log_filename=os.path.join(model_dir, 'game_script.log'))


def setup_eval_log(log_filename):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_config_file = '{}/../../../conf/logging-eval.yaml'.format(current_dir)
    setup_logging(
        default_path=log_config_file,
        local_log_filename=log_filename)
