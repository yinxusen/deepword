from collections import namedtuple
from typing import List, Dict, Tuple

import numpy as np
from albert.tokenization import FullTokenizer as AlbertTok
from bert.tokenization import FullTokenizer as BertTok
from nltk import word_tokenize

from deeptextworld.hparams import conventions
from deeptextworld.log import Logging
from deeptextworld.utils import load_vocab, get_token2idx, eprint


class Memolet(namedtuple(
    "Memolet", (
        "tid",
        "sid",
        "gid",
        "aid",
        "token_id",
        "a_len",
        "reward",
        "is_terminal",
        "action_mask",
        "sys_action_mask",
        "next_action_mask",
        "next_sys_action_mask",
        "q_actions"))):
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


class Tokenizer(object):
    @property
    def vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    @property
    def inv_vocab(self) -> Dict[int, str]:
        raise NotImplementedError()

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError()

    def de_tokenize(self, ids: List[int]) -> str:
        raise NotImplementedError()

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError()

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        raise NotImplementedError()


class NLTKTokenizer(Tokenizer):
    """
    Vocab is token2idx, inv_vocab is idx2token
    """

    def __init__(self, vocab_file, do_lower_case):
        self._special_tokens = [
            conventions.nltk_unk_token,
            conventions.nltk_padding_token,
            conventions.nltk_sos_token,
            conventions.nltk_eos_token]
        self._inv_vocab = load_vocab(vocab_file)
        if do_lower_case:
            self._inv_vocab = [
                w.lower() if w not in self._special_tokens else w
                for w in self._inv_vocab]
        self._do_lower_case = do_lower_case
        self._vocab = get_token2idx(self._inv_vocab)
        self._inv_vocab = dict([(v, k) for k, v in self._vocab.items()])
        self._unk_val_id = self._vocab[conventions.nltk_unk_token]
        self._s2c = {
            conventions.nltk_unk_token: "U",
            conventions.nltk_padding_token: "O",
            conventions.nltk_sos_token: "S",
            conventions.nltk_eos_token: "E"}
        self._c2s = dict(zip(self._s2c.values(), self._s2c.keys()))

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    def convert_tokens_to_ids(self, tokens):
        indexed = [self._vocab.get(t, self._unk_val_id) for t in tokens]
        return indexed

    def convert_ids_to_tokens(self, ids):
        tokens = [self._inv_vocab[i] for i in ids]
        return tokens

    def tokenize(self, text):
        if any([sc in text for sc in self._special_tokens]):
            new_txt = text
            for sc in self._special_tokens:
                new_txt = new_txt.replace(sc, self._s2c[sc])
            tokens = word_tokenize(new_txt)
            tokens = [self._c2s[t] if t in self._c2s else t for t in tokens]
        else:
            tokens = word_tokenize(text)

        if self._do_lower_case:
            return [
                t.lower() if t not in self._special_tokens else t
                for t in tokens]
        else:
            return tokens

    def de_tokenize(self, ids: List[int]) -> str:
        res = " ".join(
            filter(lambda t: t not in self._special_tokens,
                   self.convert_ids_to_tokens(ids)))
        return res


class BertTokenizer(Tokenizer):
    def __init__(self, vocab_file, do_lower_case):
        self.tokenizer = BertTok(vocab_file, do_lower_case)
        self._special_tokens = [
            conventions.bert_unk_token,
            conventions.bert_padding_token,
            conventions.bert_cls_token,
            conventions.bert_sep_token,
            conventions.bert_mask_token]

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def de_tokenize(self, ids):
        res = " ".join(
            filter(lambda t: t not in self._special_tokens,
                   self.convert_ids_to_tokens(ids)))
        res = res.replace(" ##", "")
        return res

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)


class AlbertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case, spm_model_file):
        super(BertTokenizer, self).__init__(vocab_file, do_lower_case)
        self.tokenizer = AlbertTok(vocab_file, do_lower_case, spm_model_file)
        self._special_tokens = [
            conventions.albert_unk_token,
            conventions.albert_padding_token,
            conventions.albert_cls_token,
            conventions.albert_sep_token,
            conventions.albert_mask_token]

    def de_tokenize(self, ids):
        res = " ".join(
            filter(lambda t: t not in self._special_tokens,
                   self.convert_ids_to_tokens(ids)))
        res = res.replace(u"\u2581", " ")
        return res


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


def pad_action(
        action_ids: List[int],
        max_size: int,
        padding_val_id: int) -> List[int]:
    if 0 < len(action_ids) < max_size:
        return action_ids + [padding_val_id] * (max_size - len(action_ids))
    else:
        return action_ids[:max_size]


def tj2ids(
        trajectory: List[ActionMaster],
        tokenizer: Tokenizer,
        with_action_padding: bool = False,
        max_action_size: int = 10,
        padding_val_id: int = 0) -> Tuple[List[int], List[int]]:
    ids = []
    master_mask = []
    for am in trajectory:
        action_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(am.action))
        if with_action_padding:
            action_ids = pad_action(action_ids, max_action_size, padding_val_id)
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
        max_action_size: int = 10
) -> Tuple[List[int], int, List[int]]:
    """
    Given trajectory (a list of ActionMaster),
    return the src and src_len as DQN input
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
        max_action_size: int = 10
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
        action_matrix: np.ndarray, action_len: np.ndarray,
        action_mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, int]]:
    valid_idx = np.where(action_mask)[0]
    inv_valid_idx = dict([(mid, i) for i, mid in enumerate(valid_idx)])
    admissible_action_matrix = action_matrix[valid_idx, :]
    admissible_action_len = action_len[valid_idx]
    return (
        admissible_action_matrix, admissible_action_len, len(valid_idx),
        inv_valid_idx)


def batch_drrn_action_input(
        action_matrices: List[np.ndarray], action_lens: List[np.ndarray],
        action_masks: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, List[int], List[Dict[int, int]]]:
    admissible_action_matrices = []
    admissible_action_lens = []
    actions_repeats = []
    group_inv_valid_idx = []
    for i in range(len(action_masks)):
        a_matrix, a_len, size, inv_valid_idx = drrn_action_input(
            action_matrices[i], action_lens[i], action_masks[i])
        admissible_action_matrices.append(a_matrix)
        admissible_action_lens.append(a_len)
        actions_repeats.append(size)
        group_inv_valid_idx.append(inv_valid_idx)
    return (
        np.concatenate(np.asarray(admissible_action_matrices), axis=0),
        np.concatenate(np.asarray(admissible_action_lens), axis=0),
        actions_repeats,
        group_inv_valid_idx)


def convert_real_id_to_group_id(
        real_id: List[int], group_inv_valid_idx: List[Dict[int, int]],
        actions_repeats: List[int]) -> List[int]:
    actions_slices = np.insert(np.cumsum(actions_repeats)[:-1], 0, 0)
    masked_id = [
        inv_idx[rid] for rid, inv_idx in zip(real_id, group_inv_valid_idx)]
    group_id = masked_id + actions_slices
    return group_id


def bert_commonsense_input(
        action_matrix: np.ndarray, action_len: np.ndarray,
        trajectory: List[int], trajectory_len: int,
        sep_val_id: int, cls_val_id: int,
        num_tokens: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given one trajectory and its admissible actions, create a training
    set of input for Bert.

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
    :return: trajectory + action; segmentation ids; sizes
    """

    assert action_matrix.ndim == 2, "action_matrix: {}".format(action_matrix)

    tj = np.concatenate([
        np.asarray([cls_val_id]), trajectory[:trajectory_len],
        np.asarray([sep_val_id])])

    n_rows, n_cols = action_matrix.shape
    tj = np.repeat(tj[None, :], n_rows, axis=0)
    seg_tj = np.zeros_like(tj, dtype=np.int)

    # make action_matrix n_cols = n_cols + k to fill in [SEP] safer
    action_matrix = np.concatenate(
        [action_matrix,
         np.zeros([n_rows, num_tokens - n_cols - trajectory_len - 2])],
        axis=-1)
    action_matrix[range(n_rows), action_len] = sep_val_id
    seg_action = np.ones_like(action_matrix, dtype=np.int)

    inp = np.concatenate([tj, action_matrix], axis=-1)
    seg_tj_action = np.concatenate([seg_tj, seg_action], axis=-1)

    # valid length plus 3 for [CLS] [SEP] and [SEP]
    inp_size = trajectory_len + action_len + 3
    return inp.astype(np.int), seg_tj_action, inp_size


def get_best_1d_action(q_actions_t, actions, mask=1):
    """
    :param q_actions_t: a q-vector of a state computed from TF at step t
    :param actions: action list
    :param mask:
    """
    action_idx, q_val = get_best_1d_q(q_actions_t, mask)
    action = actions[action_idx]
    return action_idx, q_val, action


def get_best_1d_q(q_actions_t, mask=1):
    """
    choose the action with the best q value, without choosing from inadmissible
    actions.
    Notice that it is possible q values of all admissible actions are smaller
    than zero.
    :param q_actions_t: q vector
    :param mask: integer 1 means all actions are admissible. otherwise a np
    array will be given, and each 1 means admissible while 0 not.
    :return:
    """
    mask = np.ones_like(q_actions_t) * mask
    inv_mask = np.logical_not(mask)
    min_q_val = np.min(q_actions_t)
    q_actions_t = q_actions_t * mask + min_q_val * inv_mask
    action_idx = np.argmax(q_actions_t)
    q_val = q_actions_t[action_idx]
    return action_idx, q_val


def get_batch_best_1d_idx(
        q_actions: np.ndarray, actions_repeats: List[int]) -> List[int]:
    """
    get a batch of best action index of q-values
    actions_repeats indicates how many elements are in the same group.
    e.g. q_actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    actions_repeats = [3, 4, 3]
    then q_actions can be split into three groups:
    [1, 2, 3], [4, 5, 6, 7], [8, 9, 10];
    we compute the best idx for each group
    :return:
    """
    actions_slices = np.cumsum(actions_repeats)[:-1]
    qs_slices = np.split(q_actions, actions_slices)
    actions_idx_per_slice = np.asarray([np.argmax(qs) for qs in qs_slices])
    actions_idx = np.insert(actions_slices, 0, 0) + actions_idx_per_slice
    return actions_idx


def sample_batch_1d_idx(
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
            companion = list(np.random.choice(remains, size=k-1, replace=False))
        else:
            companion = list(np.random.choice(remains, size=k-1, replace=True))
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


def get_random_1d_action(actions, mask=1):
    """
    Random sample an action but avoid choosing where mask == 0
    :param actions: action list
    :param mask: mask for the action list. 1 means OK to choose, 0 means NO.
           could be either an integer, or a numpy array the same size with
           actions.
    """
    mask = np.ones_like(actions, dtype=np.int) * mask
    action_idx = np.random.choice(np.where(mask == 1)[0])
    action = actions[action_idx]
    return action_idx, action


def get_sampled_1d_action(q_actions_t, actions, mask, temperature=1):
    """
    Choose an action w.r.t q_actions_t as logits and avoid choosing
    where mask == 0. Use temperature to control randomness.
    :param q_actions_t: logits
    :param actions: action list
    :param mask: array of mask, mask == 0 means inadmissible actions.
    :param temperature: use to control randomness, the higher the more random
    """
    q_actions_t[np.where(mask == 0)] = -np.inf
    action_idx = categorical_without_replacement(q_actions_t * temperature)
    q_val = q_actions_t[action_idx]
    action = actions[action_idx]
    return action_idx, q_val, action
