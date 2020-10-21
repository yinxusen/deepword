import glob
import os
from collections import namedtuple
from typing import List, Dict, Tuple

import numpy as np

from deepword.log import Logging
from deepword.trajectory import Trajectory
from deepword.utils import get_hash


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


class ActionMaster(object):
    def __init__(
            self, action: List[int], master: List[int]):
        self._ids = action + master
        self._lens = [len(action), len(master)]

    @property
    def ids(self):
        return self._ids

    @property
    def action_ids(self):
        return self._ids[:self._lens[0]]

    @property
    def master_ids(self):
        return self._ids[self._lens[0]:]

    @property
    def lens(self):
        return self._lens


class ObsInventory(namedtuple(
        "ObsInventory", ("obs", "inventory", "sid", "hs"))):
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

    Examples:
        >>> # suppose there are these files:
        >>> # actions-99.npz, actions-100.npz, actions-200.npz
        >>> get_path_tags("/path/to/data", "actions")
        [99, 100, 200]
    """
    all_paths = glob.glob(
        os.path.join(path, "{}-*.npz".format(prefix)), recursive=False)
    tags = list(
        map(lambda fn: int(os.path.splitext(fn)[0].split("-")[-1]),
            map(lambda p: os.path.basename(p), all_paths)))
    return tags


def drrn_action_input(
        action_matrix: np.ndarray,
        action_len: np.ndarray,
        action_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, int]]:
    """
    Select actions from `action_mask`.

    Args:
        action_matrix: action matrix for a game
        action_len: lengths for actions in the `action_matrix`
        action_mask: list of indices of selected actions

    Returns:
        selected action matrix, selected action len, number of actions selected,
         and the mapping from real ID to mask ID.
        real ID: the action index in the original `action_matrix`
        mask ID: the action index in the `action_mask`

    Examples:
        >>> a_mat = np.asarray([
        >>>     [1, 2, 3, 4, 0],
        >>>     [2, 2, 1, 3, 1],
        >>>     [3, 1, 0, 0, 0],
        >>>     [6, 9, 9, 1, 0]])
        >>> a_len = np.asarray([4, 5, 2, 4])
        >>> a_mask = np.asarray([1, 3])
        >>> drrn_action_input(a_mat, a_len, a_mask)
        [[2, 2, 1, 3, 1], [6, 9, 9, 1, 0]]
        [5, 4]
        {1: 0, 3: 1}
    """
    id_real2mask = dict([(mid, i) for i, mid in enumerate(action_mask)])
    action_matrix = action_matrix[action_mask, :]
    action_len = action_len[action_mask]
    return action_matrix, action_len, len(action_mask), id_real2mask


def batch_drrn_action_input(
        action_matrices: List[np.ndarray],
        action_lens: List[np.ndarray],
        action_masks: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, List[int], List[Dict[int, int]]]:
    """
    Select actions from `action_masks` in a batch

    see :py:func:`deepword.agents.utils.drrn_action_input`
    """
    mats, lens, actions_repeats, id_real2mask = zip(*[
        drrn_action_input(mat, l, mask) for mat, l, mask
        in zip(action_matrices, action_lens, action_masks)])
    inp_matrix = np.concatenate(mats, axis=0)
    inp_len = np.concatenate(lens, axis=0)
    return inp_matrix, inp_len, actions_repeats, id_real2mask


def id_real2batch(
        real_id: List[int], id_real2mask: List[Dict[int, int]],
        actions_repeats: List[int]) -> List[int]:
    """
    Transform real IDs to IDs in a batch

    An explanation of three ID system for actions, depending on which location
    does the action be in.

    In the action matrix of the game: real ID. E.g. a game with three actions
    `["go east", "go west", "eat meal"]`, then the real IDs are `[0, 1, 2]`

    In the action mask for each step of game-playing. E.g. when play at a
    step with admissible action as `["go east", "eat meal"]`, then the
    mask IDs are `[0, 1]`, mapping to the real IDs are `[0, 2]`.

    In a batch for training. E.g. in a batch of 2 entries, each entry is from
    a different game, say, game-1 and game-2.

    Game-1, at the step of playing, has two actions, say `[0, 2]`;

    Game-2, at the step of playing, has three actions, say, `[0, 4, 10]`.

    Supposing the agent choose action-0 from game-1 for entry-1, and
    action-4 from game-2 for entry-2. Now the real IDs are
    `[0, 4]`. However, the mask IDs are `[0, 1]`.

    Why action-4 becomes action-1? Because for that step of game-2, there
    are only three action `[0, 4, 10]`, and the action-4
    is placed at position 1.

    Converting mask IDs to batch IDs, we get `[0, 3]`.

    Why action-1 becomes action-3? Because if we place actions (mask IDs)
    for entry-1 and entry-2 together, it becomes `[[0, 1], [0, 1, 2]]`.
    The action list is then flatten into `[0, 1, 0, 1, 2]`, then re-indexed as
    `[0, 1, 2, 3, 4]`. So action-1 maps to action-3 for entry-2.

    Args:
        real_id: action ids for each game in the original action_matrix of that
         game
        id_real2mask: list of mappings from real IDs to mask IDs
        actions_repeats: action sizes in each group

    Returns:
        a list of batch IDs

    Examples:
        >>> rids = [0, 4]
        >>> id_maps = [{0: 0, 2: 1}, {0: 1, 4: 1, 10: 2}]
        >>> repeats = [2, 3]
        >>> id_real2batch(rids, id_maps, repeats)
        [0, 3]
    """
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

    # TODO: automatically reduce 3 tokens in the front
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
    """
    Find the best Q-value given a 1D Q-vector

    Args:
        q_actions: a vector of Q-values

    Returns:
        best action index, Q-value

    Examples:
        >>> q_vec = np.asarray([0.1, 0.2, 0.3, 0.4])
        >>> get_best_1d_q(q_vec)
        3, 0.4
    """
    action_idx = int(np.argmax(q_actions))
    q_val = q_actions[action_idx]
    return action_idx, q_val


def get_best_batch_ids(
        q_actions: np.ndarray, actions_repeats: List[int]) -> List[int]:
    """
    Get a batch of best action index of q-values for each group defined by
     `actions_repeats`

    Args:
        q_actions: a 1D Q-vector
        actions_repeats: groups of number of actions, indicating how many
         elements are in the same group.

    Returns:
        best action index for each group

    Examples:
        >>> q_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> repeats = [3, 4, 3]
        >>> #Q-vector splits into three groups containing 3, 4, 3 Q-values
        >>> # shaded_qs = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
        >>> get_best_batch_ids(np.asarray(q_vec), repeats)
        [3, 7, 10]
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


def get_hash_state(obs: str, inv: str) -> str:
    """
    Generate hash state from observation and inventory
    Args:
        obs: observation of current step
        inv: inventory of current step

    Returns:
        hash state of current step
    """
    return get_hash(obs + "\n" + inv)


def get_snn_keys(
        hash_states2tjs: Dict[str, Dict[int, List[int]]],
        tjs: Trajectory,
        size: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]],
           List[Tuple[int, int]]]:
    """
    Get SNN training pairs from trajectories.

    Args:
        hash_states2tjs: the mapping from hash state to trajectory
        tjs: the trajectories
        size: batch size

    Returns:
        target_set, same_set and diff_set
        each set contains keys of (tid, sid) to locate trajectory
    """

    non_empty_keys = list(
        filter(lambda x: hash_states2tjs[x] != {},
               hash_states2tjs.keys()))
    perm_keys = list(np.random.permutation(non_empty_keys))
    state_pairs = list(zip(non_empty_keys, perm_keys))

    target_set = []
    same_set = []
    diff_set = []

    i = 0
    while i < size:
        for j, (sk1, sk2) in enumerate(state_pairs):
            if sk1 == sk2:
                sk2 = non_empty_keys[(j + 1) % len(non_empty_keys)]

            try:
                tid_pair = np.random.choice(
                    list(hash_states2tjs[sk1].keys()),
                    size=2, replace=False)
            except ValueError:
                tid_pair = list(hash_states2tjs[sk1].keys()) * 2

            target_tid = tid_pair[0]
            same_tid = tid_pair[1]

            if (target_tid == same_tid and
                    len(hash_states2tjs[sk1][same_tid]) == 1):
                continue

            diff_tid = np.random.choice(
                list(hash_states2tjs[sk2].keys()), size=None)

            # remove empty trajectory
            if (target_tid not in tjs.trajectories
                    or same_tid not in tjs.trajectories
                    or diff_tid not in tjs.trajectories):
                continue

            target_sid = np.random.choice(
                list(hash_states2tjs[sk1][target_tid]), size=None)
            same_sid = np.random.choice(
                list(hash_states2tjs[sk1][same_tid]), size=None)
            diff_sid = np.random.choice(
                list(hash_states2tjs[sk2][diff_tid]), size=None)

            target_set.append((target_tid, target_sid))
            same_set.append((same_tid, same_sid))
            diff_set.append((diff_tid, diff_sid))

            i += 1
            if i >= size:
                break

    return target_set, same_set, diff_set
