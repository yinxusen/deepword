import glob
import os
from collections import namedtuple
from typing import List, Dict, Tuple, Optional, Iterator

import numpy as np

from deeptextworld.log import Logging
from deeptextworld.tokenizers import Tokenizer


class Memolet(namedtuple(
    "Memolet", (
            "tid",
            "sid",
            "gid",
            "aid",
            "token_id",
            "a_len",
            "a_type",
            "reward",
            "is_terminal",
            "action_mask",
            "sys_action_mask",
            "next_action_mask",
            "next_sys_action_mask",
            "q_actions"))):
    """
    end_of_episode: game stops by 1) winning, 2) losing, or 3) exceeding
    maximum number of steps.
    is_terminal: is current step reaches the terminal game state by winning
    or losing. is_terminal = True means for the current step, q value equals
    to the instant reward.

    TODO: Notice that end_of_episode doesn't imply is_terminal. Only winning
        or losing means is_terminal = True.
    """
    pass


class ActionMaster(namedtuple("ActionMaster", ("action", "master"))):
    pass


class ObsInventory(namedtuple("ObsInventory", ("obs", "inventory"))):
    pass


class ActionDesc(namedtuple(
    "ActionDesc", (
            "action_type",
            "action_idx",
            "token_idx",
            "action_len",
            "action",
            "q_actions"))):
    pass


class GenSummary(namedtuple(
        "GenSummary", ("ids", "tokens", "gens", "q_action", "len"))):

    def __repr__(self):
        return (" ".join(["{}[{:.2f}]".format(t, p)
                          for t, p in zip(self.tokens, self.gens)])
                + "\t{}".format(self.q_action))


class CommonActs(namedtuple(
    "CommonActs",
    ("examine_cookbook", "prepare_meal", "eat_meal", "look", "inventory",
     "gn", "gs", "ge", "gw"))):
    pass


ACT = CommonActs(
    examine_cookbook="examine cookbook",
    prepare_meal="prepare meal",
    eat_meal="eat meal",
    look="look",
    inventory="inventory",
    gn="go north",
    gs="go south",
    ge="go east",
    gw="go west")


class ActType(namedtuple(
    "ActType",
    ("rnd", "rule", "rnd_walk", "policy_drrn", "policy_gen",
     "jitter", "policy_tbl"))):
    pass


ACT_TYPE = ActType(
    rnd="random_choose_action",
    rule="rule_based_action",
    rnd_walk="random_walk_action",
    policy_drrn="drrn_action",
    policy_gen="gen_action",
    jitter="jitter_action",
    policy_tbl="tabular_action")


class EnvInfosKey(namedtuple(
    "KeyInfo",
    ("recipe", "desc", "inventory", "max_score", "won", "lost",
     "actions", "templates", "verbs", "entities"))):
    pass


INFO_KEY = EnvInfosKey(
    recipe="extra.recipe",
    desc="description",
    inventory="inventory",
    max_score="max_score",
    won="won",
    lost="lost",
    actions="admissible_commands",
    templates="command_templates",
    verbs="verbs",
    entities="entities")


class ScheduledEPS(Logging):
    def eps(self, t):
        raise NotImplementedError()


class LinearDecayedEPS(ScheduledEPS):
    def __init__(self, decay_step, init_eps=1, final_eps=0):
        super(LinearDecayedEPS, self).__init__()
        self.init_eps = init_eps
        self.final_eps = final_eps
        self.decay_step = decay_step
        self.decay_speed = (
                1. * (self.init_eps - self.final_eps) / self.decay_step)

    def eps(self, t):
        if t < 0:
            return self.init_eps
        eps_t = max(self.init_eps - self.decay_speed * t, self.final_eps)
        return eps_t


class ScannerDecayEPS(ScheduledEPS):
    def __init__(
            self, decay_step, decay_range,
            next_init_eps_rate=0.8, init_eps=1, final_eps=0):
        super(ScannerDecayEPS, self).__init__()
        self.init_eps = init_eps
        self.final_eps = final_eps
        self.decay_range = decay_range
        self.n_ranges = decay_step // decay_range
        self.range_init = list(map(
            lambda i: max(init_eps * (next_init_eps_rate ** i), 0.3),
            range(self.n_ranges)))
        self.decay_speed = list(map(
            lambda es: 1. * (es - self.final_eps) / self.decay_range,
            self.range_init))

    def eps(self, t):
        if t < 0:
            return self.init_eps
        range_idx = t // self.decay_range
        range_t = t % self.decay_range
        if range_idx >= self.n_ranges:
            return self.final_eps
        eps_t = (self.range_init[range_idx]
                 - range_t * self.decay_speed[range_idx])
        self.debug("{} - {} - {} - {} - {}".format(
            range_idx, range_t, self.range_init[range_idx],
            self.decay_speed[range_idx], eps_t))
        return eps_t


def get_path_tags(path: str, prefix: str) -> List[int]:
    """
    Get tag from a path of saved objects. E.g. actions-100.npz
    100 will be extracted
    Make sure the item to be extracted is saved with suffix of npz.

    Args:
        path: path to find files with prefix
        prefix: prefix

    Returns:
        list of all tags
    """
    all_paths = glob.glob(
        os.path.join(path, "{}-*.npz".format(prefix)), recursive=False)
    tags = list(
        map(lambda fn: int(os.path.splitext(fn)[0].split("-")[-1]),
            map(lambda p: os.path.basename(p), all_paths)))
    return tags


def align_batch_str(
        ids: List[List[int]], str_len_allowance: int,
        padding_val_id: int, valid_len: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a batch of string indexes.
    The maximum length will not exceed str_len_allowance.
    Each array of ids will be either padded or trimmed to reach
    the maximum length, notice that padded length won't be counted in as valid
    length.

    Args:
        ids: a list of array of index (int)
        str_len_allowance:
        padding_val_id:
        valid_len:

    Returns:
        aligned ids and aligned length
    """
    def align() -> Iterator[Tuple[List[int], int]]:
        m = min(str_len_allowance, np.max(valid_len))
        for s, l in zip(ids, valid_len):
            if 0 <= l < m:
                yield s[:l] + [padding_val_id] * (m - l), l
            else:
                yield s[:m], m

    aligned_ids, aligned_len = zip(*align())
    return np.asarray(aligned_ids), np.asarray(aligned_len)


def tj2ids(
        trajectory: List[ActionMaster],
        tokenizer: Tokenizer,
        with_action_padding: bool = False,
        max_action_size: Optional[int] = None,
        padding_val_id: int = 0) -> Tuple[List[int], List[int]]:
    """
    Convert a trajectory (list of ActionMaster) into ids
    Compute segmentation ids for masters (1) and actions (0)
    pad actions if required.
    """
    ids = []
    master_mask = []
    for am in trajectory:
        action_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(am.action))
        if with_action_padding:
            if len(action_ids) < max_action_size:
                action_ids += [padding_val_id] * (
                        max_action_size - len(action_ids))
            else:
                action_ids = action_ids[:max_action_size]
        master_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(am.master))
        ids += action_ids
        ids += master_ids
        master_mask += [0] * len(action_ids)
        master_mask += [1] * len(master_ids)
    return ids, master_mask


def dqn_input(
        trajectory: List[ActionMaster],
        tokenizer: Tokenizer,
        num_tokens: int,
        padding_val_id: int,
        with_action_padding: bool = False,
        max_action_size: Optional[int] = None
) -> Tuple[List[int], int, List[int]]:
    """
    Given a trajectory (a list of ActionMaster), get trajectory indexes, length
    and master mask (master marked as 1 while action marked as 0).
    Pad the trajectory to num_tokens.
    Pad actions if required.
    """
    trajectory_ids, raw_master_mask = tj2ids(
        trajectory, tokenizer,
        with_action_padding, max_action_size, padding_val_id)
    padding_size = num_tokens - len(trajectory_ids)
    if padding_size >= 0:
        src = trajectory_ids + [padding_val_id] * padding_size
        master_mask = raw_master_mask + [0] * padding_size
        src_len = len(trajectory_ids)
    else:
        src = trajectory_ids[-padding_size:]
        master_mask = raw_master_mask[-padding_size:]
        src_len = num_tokens
    return src, src_len, master_mask


def batch_dqn_input(
        trajectories: List[List[ActionMaster]],
        tokenizer: Tokenizer,
        num_tokens: int,
        padding_val_id: int,
        with_action_padding: bool = False,
        max_action_size: Optional[int] = None
) -> Tuple[List[List[int]], List[int], List[List[int]]]:
    batch_src = []
    batch_src_len = []
    batch_mask = []
    for tj in trajectories:
        src, src_len, master_mask = dqn_input(
            tj, tokenizer, num_tokens, padding_val_id,
            with_action_padding, max_action_size)
        batch_src.append(src)
        batch_src_len.append(src_len)
        batch_mask.append(master_mask)
    return batch_src, batch_src_len, batch_mask


def drrn_action_input(
        action_matrix: np.ndarray,
        action_len: np.ndarray,
        action_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, int]]:
    id_real2mask = dict([(mid, i) for i, mid in enumerate(action_mask)])
    action_matrix = action_matrix[action_mask, :]
    action_len = action_len[action_mask]
    return action_matrix, action_len, len(action_mask), id_real2mask


def batch_drrn_action_input(
        action_matrices: List[np.ndarray],
        action_lens: List[np.ndarray],
        action_masks: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, List[int], List[Dict[int, int]]]:
    inp = [
        drrn_action_input(mat, l, mask) for mat, l, mask
        in zip(action_matrices, action_lens, action_masks)]
    inp_matrix = np.concatenate([x[0] for x in inp], axis=0)
    inp_len = np.concatenate([x[1] for x in inp], axis=0)
    actions_repeats = [x[2] for x in inp]
    id_real2mask = [x[3] for x in inp]
    return inp_matrix, inp_len, actions_repeats, id_real2mask


def id_real2batch(
        real_id: List[int], id_real2mask: List[Dict[int, int]],
        actions_repeats: List[int]) -> List[int]:
    batch_id = [0] + list(np.cumsum(actions_repeats)[:-1])
    return [inv_id[rid] + bid for rid, inv_id, bid
            in zip(real_id, id_real2mask, batch_id)]


def bert_commonsense_input(
        action_matrix: np.ndarray, action_len: np.ndarray,
        trajectory: List[int], trajectory_len: int,
        sep_val_id: int, cls_val_id: int,
        num_tokens: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given one trajectory and its admissible actions, create a training
    set of input for Bert.

    Notice: the trajectory_len and action_len need to be confirmed that to have
    special tokens e.g. [CLS], [SEP] positions to be reserved.

    E.g. input: [1, 2, 3], and action_matrix [[1, 3], [2, PAD], [4, PAD]]
    suppose we need length to be 10.
    output:
      [[CLS, 1, 2, 3, SEP, 1, 3,   SEP, PAD, PAD, PAD],
       [CLS, 1, 2, 3, SEP, 2, SEP, PAD, PAD, PAD, PAD],
       [CLS, 1, 2, 3, SEP, 4, SEP, PAD, PAD, PAD, PAD]]
    segment of trajectory and actions:
    [[0, 0, 0, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 1, 1, 0]]
    input size:
    [8, 7, 7]

    Returns:
        trajectory + action; segmentation ids; sizes
    """

    assert action_matrix.ndim == 2, "action_matrix: {}".format(action_matrix)
    assert np.all(trajectory_len + action_len <= num_tokens - 3), \
        "trajectory len or action len are too large"

    tj = np.concatenate([
        np.asarray([cls_val_id]), trajectory[:trajectory_len],
        np.asarray([sep_val_id])])

    n_rows, n_cols = action_matrix.shape
    tj = np.repeat(tj[None, :], n_rows, axis=0)
    seg_tj = np.zeros_like(tj, dtype=np.int32)

    n_cols_tj = tj.shape[1]
    n_cols_action = num_tokens - n_cols_tj
    n_cols_padding = n_cols_action - n_cols

    action_matrix = np.concatenate(
        [action_matrix, np.zeros([n_rows, n_cols_padding])], axis=-1)
    action_matrix[range(n_rows), action_len] = sep_val_id
    seg_action = np.ones_like(action_matrix, dtype=np.int32)

    inp = np.concatenate([tj, action_matrix], axis=-1)
    seg_tj_action = np.concatenate([seg_tj, seg_action], axis=-1)

    # valid length plus 3 for [CLS] [SEP] and [SEP]
    inp_size = trajectory_len + action_len + 3
    return inp.astype(np.int32), seg_tj_action, inp_size


def get_best_1d_q(q_actions: np.ndarray) -> Tuple[int, float]:
    action_idx = int(np.argmax(q_actions))
    q_val = q_actions[action_idx]
    return action_idx, q_val


def get_best_batch_ids(
        q_actions: np.ndarray, actions_repeats: List[int]) -> List[int]:
    """
    get a batch of best action index of q-values
    actions_repeats indicates how many elements are in the same group.
    e.g. q_actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    actions_repeats = [3, 4, 3]
    then q_actions can be split into three groups:
    [1, 2, 3], [4, 5, 6, 7], [8, 9, 10];
    we compute the best idx for each group
    """
    assert q_actions.ndim == 1
    assert np.all(np.greater(actions_repeats, 0))
    actions_slices = np.cumsum(actions_repeats)[:-1]
    qs_slices = np.split(q_actions, actions_slices)
    actions_idx_per_slice = np.asarray([np.argmax(qs) for qs in qs_slices])
    actions_idx = np.insert(actions_slices, 0, 0) + actions_idx_per_slice
    return actions_idx


def sample_batch_ids(
        q_actions: np.ndarray, actions_repeats: List[int], k: int) -> List[int]:
    """
    get a batch of sampled action index of q-values
    actions_repeats indicates how many elements are in the same group.
    e.g. q_actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    actions_repeats = [3, 4, 3]
    then q_actions can be split into three groups:
    [1, 2, 3], [4, 5, 6, 7], [8, 9, 10];

    we sample from the indexes, we get the best idx in each group as the first
    one in that group, then sample another k - 1 elements for each group.
    If the number of elements in that group smaller than k - 1, we choose sample
    with replacement.
    """
    assert np.all(np.greater(actions_repeats, 1)), \
        "actions_repeats should greater than one"
    batch_size = len(actions_repeats)
    actions_slices = np.cumsum(actions_repeats)[:-1]
    qs_slices = np.split(q_actions, actions_slices)
    action_idx_blocks_per_slice = []
    for blk_i in range(batch_size):
        curr_best = int(np.argmax(qs_slices[blk_i]))
        remains = (list(range(curr_best)) +
                   list(range(curr_best + 1, len(qs_slices[blk_i]))))
        if len(remains) >= k - 1:
            companion = list(
                np.random.choice(remains, size=k - 1, replace=False))
        else:
            companion = list(
                np.random.choice(remains, size=k - 1, replace=True))
        action_idx_blocks_per_slice.append([curr_best] + companion)
    actions_idx = (
            np.insert(actions_slices, 0, 0)[:, np.newaxis] +
            np.asarray(action_idx_blocks_per_slice)).reshape((batch_size * k))
    return actions_idx


def categorical_without_replacement(logits, k=1):
    """
    Courtesy of https://github.com/tensorflow/tensorflow/issues/\
    9260#issuecomment-437875125
    also cite here:
    @misc{vieira2014gumbel,
        title = {Gumbel-max trick and weighted reservoir sampling},
        author = {Tim Vieira},
        url = {http://timvieira.github.io/blog/post/2014/08/01/\
        gumbel-max-trick-and-weighted-reservoir-sampling/},
        year = {2014}
    }
    Notice that the logits represent unnormalized log probabilities,
    in the citation above, there is no need to normalized them first to add
    the Gumbel random variant, which surprises me! since I thought it should
    be `logits - tf.reduce_logsumexp(logits) + z`
    """
    z = -np.log(-np.log(np.random.uniform(0, 1, logits.shape)))
    return np.argsort(logits + z)[:k] if k > 1 else np.argmax(logits + z)


def get_action_idx_pair(
        action_matrix: np.ndarray, action_len: np.ndarray, sos_id: int,
        eos_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create action index pair for seq2seq training.
    Given action index, e.g. [1, 2, 3, 4, pad, pad, pad, pad],
    with 0 as sos_id, and -1 as eos_id,
    we create training pair: [0, 1, 2, 3, 4, pad, pad, pad]
    as the input sentence, and [1, 2, 3, 4, -1, pad, pad, pad]
    as the output sentence.

    Notice that we remove the final pad to keep the action length unchanged.
    Notice 2. pad should be indexed as 0.

    Args:
        action_matrix: np array of action index of N * K, there are N,
         and each of them has a length of K (with paddings).
        action_len: length of each action (remove paddings).
        sos_id:
        eos_id:

    Returns:
        action index as input, action index as output, new action len
    """
    n_rows, max_col_size = action_matrix.shape
    action_id_in = np.concatenate(
        [np.full((n_rows, 1), sos_id), action_matrix[:, :-1]], axis=1)
    # make sure original action_matrix is untouched.
    action_id_out = np.copy(action_matrix)
    new_action_len = np.min(
        [action_len + 1, np.zeros_like(action_len) + max_col_size], axis=0)
    action_id_out[list(range(n_rows)), new_action_len - 1] = eos_id
    return action_id_in, action_id_out, new_action_len


def remove_zork_version_info(text):
    zork_info = [
        "ZORK I: The Great Underground Empire",
        "Copyright (c) 1981, 1982, 1983 Infocom, Inc. All rights reserved.",
        "ZORK is a registered trademark of Infocom, Inc.",
        "Revision 88 / Serial number 840726"]
    # don't strip texts, keep the raw response
    return "\n".join(
        filter(lambda s: s.strip() not in zork_info, text.split("\n")))
