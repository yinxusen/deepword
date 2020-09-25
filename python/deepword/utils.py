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
from os import path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import ruamel.yaml
from bitarray import bitarray
from tabulate import tabulate


def get_hash(txt: str) -> str:
    """
    get hex hash value for a string
    """
    return hashlib.md5(txt.encode("utf-8")).hexdigest()


def eprint(*args, **kwargs):
    """
    print to stderr
    """
    print(*args, file=sys.stderr, **kwargs)


def flatmap(f, items):
    """
    flatmap for python
    """
    return list(chain.from_iterable(map(f, items)))


def flatten(items):
    """
    flatten a list of lists to a list
    """
    return list(chain.from_iterable(items))


def uniq(lst):
    """
    order-preserving unique
    """
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]


def load_uniq_lines(fname: str) -> List[str]:
    """
    Load unique lines from a file, line order preserved
    """
    with open(fname, 'r') as f:
        lines = map(lambda l: l.strip(), f.readlines())
    return uniq(lines)


def load_vocab(vocab_file: str) -> List[str]:
    """
    Load unique words from a vocabulary
    """
    return load_uniq_lines(vocab_file)


def load_actions(action_file: str) -> List[str]:
    """
    Load unique actions from an action file
    """
    return load_uniq_lines(action_file)


def get_token2idx(tokens: List[str]) -> Dict[str, int]:
    """
    From a list of tokens to a dict of token to position
    """
    uniq_tokens = uniq(tokens)
    return dict(map(
        lambda idx_token: (idx_token[1], idx_token[0]), enumerate(uniq_tokens)))


def ctime() -> int:
    """
    current time in millisecond
    """
    return int(round(time.time() * 1000))


def setup_logging(
        default_path: str = 'logging.yaml',
        default_level: int = logging.INFO,
        env_key: str = 'LOG_CFG',
        local_log_filename: Optional[str] = None) -> None:

    """
    Setup logging for python project

    Load YAML config file from `default_path`, or from the environment variable
    set by `env_key`. Falls back to default config if file not exist.

    if `local_log_filename` set, add a local rotating log file.
    """

    config_path = default_path
    value = os.getenv(env_key, None)
    if value:
        config_path = value
    if path.exists(config_path):
        with open(config_path, 'rt') as f:
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


def model_name2clazz(name: str):
    """
    Find the class given the model name in this package.

    Args:
        name: Model name from :py:mod:`deepword.models`

    Returns:
        the class w.r.t. the model name
    """

    from deepword.models import dqn_model, drrn_model, dsqn_model, \
        gen_model, commonsense_model, sentence_model

    for namespace in [dqn_model, drrn_model, dsqn_model, gen_model,
                      commonsense_model, sentence_model]:
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise ValueError("{} not found in models".format(name))


def learner_name2clazz(name: str):
    """
    Find the class given the learner name in this package.

    Args:
        name: Learner name from :py:mod:`deepword.students`

    Returns:
        the class w.r.t. the learner name
    """

    from deepword.students import student_learner, bert_swag, sentence_learner

    for namespace in [student_learner, bert_swag, sentence_learner]:
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise ValueError("{} not found in student learners".format(name))


def agent_name2clazz(name: str):
    """
    Find the class given the agent name in this package.

    Args:
        name: Agent name from :py:mod:`deepword.agents`

    Returns:
        the class w.r.t. the agent name
    """

    from deepword.agents import base_agent, dsqn_agent, \
        gen_agent, competition_agent, gen_drrn_agent, \
        zork_agent

    for namespace in [base_agent, dsqn_agent, gen_agent,
                      competition_agent, gen_drrn_agent,
                      zork_agent]:
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise ValueError("{} not found in agents".format(name))


def core_name2clazz(name: str):
    """
    Find the class given the core name in this package.

    Args:
        name: Agent name from :py:mod:`deepword.agents.cores`

    Returns:
        the class w.r.t. the core name
    """

    from deepword.agents import cores

    if hasattr(cores, name):
        return getattr(cores, name)
    raise ValueError("{} not found in agents".format(name))


def split_train_dev(
        game_files: List[str], train_ratio: float = 0.9, rnd_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split train/dev sets from given game files
    sort - shuffle w/ Random(42) - split

    Args:
        game_files: game files
        train_ratio: the percentage of training files
        rnd_seed: for randomly shuffle files, default = 42

    Returns:
        train_games, dev_games

    Exception:
        empty game_files
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


def load_game_files(game_path: str, f_games: Optional[str] = None) -> List[str]:
    """
    Load a dir of games, or a single game.
    if game_path represents a file, then return a list of the file;
    if game_path is a dir, then return a list of files in the dir suffixed with
      .ulx;
    if f_games is set, then load files in the game_path with names listed in
      f_games.

    Args:
        game_path: a dir, or a single file
        f_games: a file of game names, without suffix, default suffix .ulx

    Returns:
        a list of game files
    """

    if path.isfile(game_path):
        game_files = [game_path]
    elif path.isdir(game_path):
        if f_games is not None:
            with open(f_games, "r") as f:
                selected_games = map(lambda x: x.strip(), f.readlines())
            game_files = list(map(
                lambda x: path.join(game_path, "{}.ulx".format(x)),
                selected_games))
        else:
            game_files = glob.glob(path.join(game_path, "*.ulx"))
    else:
        raise ValueError("game path {} doesn't exist".format(game_path))
    return game_files


def load_and_split(game_path: str, f_games: str) -> Tuple[List[str], List[str]]:
    """
    Load games and split train dev set

    Args:
        game_path: game dir
        f_games: a file with list of games, each game name per line, without the
        suffix of ulx

    Returns:
        train_games, dev_games
    """

    game_files = load_game_files(game_path, f_games)
    train_games, dev_games = split_train_dev(game_files)
    return train_games, dev_games


dir_path = path.dirname(path.realpath(__file__))
project_path = path.join(dir_path, "../..")
fn_log = path.join(project_path, "conf/logging.yaml")
fn_log_eval = path.join(project_path, "conf/logging-eval.yaml")


def setup_train_log(model_dir: str):
    """
    Setup log for training by putting a `game_script.log` in `model_dir`.
    """
    assert path.isfile(fn_log)
    setup_logging(
        default_path=fn_log,
        local_log_filename=path.join(model_dir, 'game_script.log'))


def setup_eval_log(log_filename: str):
    """
    Setup log for evaluation

    Args:
        log_filename: the path to log file
    """
    assert path.isfile(fn_log_eval)
    setup_logging(default_path=fn_log_eval, local_log_filename=log_filename)


def bytes2idx(byte_mask: List[bytes], size: int) -> np.ndarray:
    """
    load a list of bytes to choose `1` for selected actions

    Args:
        byte_mask: a list of bytes
        size: the size of total actions

    Returns:
        an np array of indices
    """
    bit_mask = bitarray(endian='little')
    bit_mask.frombytes(byte_mask)
    bit_mask = bit_mask[:size]
    bit_mask[-1] = False
    np_mask = np.asarray(bit_mask.tolist(), dtype=np.int32)
    return np.where(np_mask == 1)[0]


def softmax(x: np.ndarray) -> np.ndarray:
    """
    numerical stability softmax
    """
    e_x = np.exp(x - np.sum(x))
    return e_x / np.sum(e_x)


def report_status(lst_of_status: List[Tuple[str, Any]]) -> str:
    """
    Pretty print a series of k-v pairs

    Args:
        lst_of_status: A list of k-v pairs

    Returns:
        a string to print
    """
    return "\n" + tabulate(lst_of_status, tablefmt="plain") + "\n"
