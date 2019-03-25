from __future__ import print_function

import os
import logging
import logging.config
import logging.handlers
import yaml
import sys
from itertools import chain
import time


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
        return list(map(lambda t: t.lower(), load_uniq_lines(vocab_file)))


def load_actions(action_file):
        return load_uniq_lines(action_file)


def get_token2idx(tokens):
    uniq_tokens = uniq(tokens)
    return dict(map(lambda idx_token: (idx_token[1], idx_token[0]), enumerate(uniq_tokens)))


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
            config = yaml.safe_load(f.read())
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
