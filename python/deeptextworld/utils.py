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
from os.path import join as pjoin
from typing import List, Tuple

import numpy as np
import ruamel.yaml
from bitarray import bitarray


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
        gen_model, commonsense_model

    for namespace in [dqn_model, drrn_model, dsqn_model, gen_model,
                      commonsense_model]:
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
    from deeptextworld.agents import base_agent, dsqn_agent, \
        gen_agent, competition_agent, gen_drrn_agent, \
        zork_agent

    for namespace in [base_agent, dsqn_agent, gen_agent,
                      competition_agent, gen_drrn_agent,
                      zork_agent]:
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise ValueError("{} not found in agents".format(name))


def core_name2clazz(name):
    from deeptextworld.agents import cores

    if hasattr(cores, name):
        return getattr(cores, name)
    raise ValueError("{} not found in agents".format(name))


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


def load_game_files(game_path, f_games=None):
    """
    Load a dir of games, or a single game.
    if game_path represents a file, then return a list of the file;
    if game_path is a dir, then return a list of files in the dir suffixed with
      .ulx;
    if f_games is set, then load files in the game_path with names listed in
      f_games.
    :param game_path: a dir, or a single file
    :param f_games: a file of game names, without suffix, default suffix .ulx
    :return: a list of game files
    """
    if os.path.isfile(game_path):
        game_files = [game_path]
    elif os.path.isdir(game_path):
        if f_games is not None:
            with open(f_games, "r") as f:
                selected_games = map(lambda x: x.strip(), f.readlines())
            game_files = list(map(
                lambda x: pjoin(game_path, "{}.ulx".format(x)),
                selected_games))
        else:
            game_files = glob.glob(pjoin(game_path, "*.ulx"))
    else:
        raise ValueError("game path {} doesn't exist".format(game_path))
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


dir_path = os.path.dirname(os.path.realpath(__file__))
project_path = pjoin(dir_path, "../..")
fn_log = pjoin(project_path, "conf/logging.yaml")
fn_log_eval = pjoin(project_path, "conf/logging-eval.yaml")


def setup_train_log(model_dir):
    assert os.path.isfile(fn_log)
    setup_logging(
        default_path=fn_log,
        local_log_filename=pjoin(model_dir, 'game_script.log'))


def setup_eval_log(log_filename):
    assert os.path.isfile(fn_log_eval)
    setup_logging(default_path=fn_log_eval, local_log_filename=log_filename)


def bytes2array(byte_action_masks):
    """
    Convert a list of byte-array masks to a list of np-array masks.
    TODO: last bit set as False to represent the end of the bit-string, i.e.
        '\0' in c/c++.
    """
    vec_action_masks = []
    for mask in byte_action_masks:
        bit_mask = bitarray(endian='little')
        bit_mask.frombytes(mask)
        bit_mask[-1] = False
        vec_action_masks.append(bit_mask.tolist())
    return np.asarray(vec_action_masks, dtype=np.int32)


def bytes2idx(byte_mask: List[bytes], size: int) -> np.ndarray:
    bit_mask = bitarray(endian='little')
    bit_mask.frombytes(byte_mask)
    bit_mask = bit_mask[:size]
    bit_mask[-1] = False
    np_mask = np.asarray(bit_mask.tolist(), dtype=np.int32)
    return np.where(np_mask == 1)[0]


def softmax(x):
    """numerical stability softmax"""
    e_x = np.exp(x - np.sum(x))
    return e_x / np.sum(e_x)


def report_status(lst_of_status: List[Tuple[str, object]]) -> str:
    return ', '.join(
        map(lambda k_v: '{}: {}'.format(k_v[0], k_v[1]), lst_of_status))
