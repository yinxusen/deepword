import glob
import math
import os
import random

import numpy as np

from deeptextworld.stats import mean_confidence_interval
from deeptextworld.utils import setup_logging


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


def load_and_split(game_path, f_games):
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


def get_action_idx_pair(action_matrix, action_len, sos_id, eos_id):
    """
    Create action index pair for seq2seq training.
    Given action index, e.g. [1, 2, 3, 4, pad, pad, pad, pad],
    with 0 as sos_id, and -1 as eos_id,
    we create training pair: [0, 1, 2, 3, 4, pad, pad, pad]
    as the input sentence, and [1, 2, 3, 4, -1, pad, pad, pad]
    as the output sentence.

    Notice that we remove the final pad to keep the action length unchanged.
    Notice 2. pad should be indexed as 0.

    :param action_matrix: np array of action index of N * K, there are N
    actions, and each of them has a length of K (with paddings).
    :param action_len: length of each action (remove paddings).
    :param sos_id:
    :param eos_id:
    :return: action index as input, action index as output, new action len
    """
    n_rows, max_col_size = action_matrix.shape
    action_id_in = np.concatenate(
        [np.full((n_rows, 1), sos_id), action_matrix[:, :-1]], axis=1)
    # make sure original action_matrix is untouched.
    action_id_out = np.copy(action_matrix)
    new_action_len = np.min(
        [action_len + 1, np.zeros_like(action_len) + max_col_size], axis=0)
    action_id_out[list(range(n_rows)), new_action_len-1] = eos_id
    return action_id_in, action_id_out, new_action_len


def test():
    action_matrix = np.random.randint(1, 10, (5, 4))
    action_len = np.random.randint(1, 5, 5)
    sos_id = 0
    eos_id = -1
    action_id_in, action_id_out, action_len = get_action_idx_pair(
        action_matrix, action_len, sos_id, eos_id)
    print(action_matrix)
    print(action_id_in)
    print(action_id_out)
    print(action_len)


if __name__ == '__main__':
    test()
