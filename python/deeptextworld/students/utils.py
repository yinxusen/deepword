import os

import numpy as np

from deeptextworld.utils import setup_logging


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
