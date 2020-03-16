from collections import namedtuple
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
from nltk import word_tokenize
from bert.tokenization import FullTokenizer as BertTok
from albert.tokenization import FullTokenizer as AlbertTok

from deeptextworld.log import Logging
from deeptextworld.utils import load_vocab, get_token2idx, flatten


@dataclass(frozen=True)
class Memolet:
    tid: int
    sid: int
    gid: str
    aid: int
    token_id: np.ndarray
    a_len: int
    reward: float
    is_terminal: bool
    action_mask: bytes
    sys_action_mask: bytes
    next_action_mask: bytes
    next_sys_action_mask: bytes
    q_actions: Optional[np.ndarray]


@dataclass(frozen=True)
class ActionMaster:
    action: str
    master: str


@dataclass(frozen=True)
class ObsInventory:
    obs: str
    inventory: str


@dataclass(frozen=True)
class ActionDesc:
    action_type: str
    action_idx: Optional[int]
    token_idx: Optional[np.ndarray]
    action_len: Optional[int]
    action: Optional[str]
    q_actions: Optional[np.ndarray]


class Tokenizer(object):
    @property
    def vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    @property
    def inv_vocab(self) -> Dict[int, str]:
        raise NotImplementedError()

    def tokenize(self, text: str) -> List[str]:
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
        self._special_chars = ["[UNK]", "[PAD]", "<S>", "</S>"]
        self._inv_vocab = load_vocab(vocab_file)
        if do_lower_case:
            self._inv_vocab = [
                w.lower() if w not in self._special_chars else w
                for w in self._inv_vocab]
        self._do_lower_case = do_lower_case
        self._vocab = get_token2idx(self._inv_vocab)
        self._inv_vocab = dict([(v, k) for k, v in self._vocab.items()])
        self._unk_val_id = self._vocab["[UNK]"]
        self._s2c = {"[UNK]": "U", "[PAD]": "O", "<S>": "S", "</S>": "E"}
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
        if any([sc in text for sc in self._special_chars]):
            new_txt = text
            for sc in self._special_chars:
                new_txt = new_txt.replace(sc, self._s2c[sc])
            tokens = word_tokenize(new_txt)
            tokens = [self._c2s[t] if t in self._c2s else t for t in tokens]
        else:
            tokens = word_tokenize(text)

        if self._do_lower_case:
            return [
                t.lower() if t not in self._special_chars else t
                for t in tokens]
        else:
            return tokens


class BertTokenizer(Tokenizer):
    def __init__(self, vocab_file, do_lower_case):
        self.tokenizer = BertTok(vocab_file, do_lower_case)

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

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)


class AlbertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case, spm_model_file):
        super(BertTokenizer, self).__init__(vocab_file, do_lower_case)
        self.tokenizer = AlbertTok(vocab_file, do_lower_case, spm_model_file)


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


def action_master2str(trajectory: List[ActionMaster]) -> List[str]:
    return flatten([[x.action, x.master] for x in trajectory])


def dqn_input(
        trajectory: Union[List[str], List[ActionMaster]],
        tokenizer: Tokenizer,
        num_tokens: int,
        padding_val_id: int) -> Tuple[List[int], int]:
    """
    Given trajectory (a list of ActionMaster),
    return the src and src_len as DQN input
    """
    if isinstance(trajectory[0], ActionMaster):
        trajectory = action_master2str(trajectory)
    trajectory = " ".join(trajectory)

    trajectory_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(trajectory))
    padding_size = num_tokens - len(trajectory_ids)
    if padding_size >= 0:
        src = trajectory_ids + [padding_val_id] * padding_size
        src_len = len(trajectory_ids)
    else:
        src = trajectory_ids[-padding_size:]
        src_len = num_tokens
    return src, src_len


def batch_dqn_input(
        trajectories: Union[List[List[str]], List[List[ActionMaster]]],
        tokenizer: Tokenizer,
        num_tokens: int,
        padding_val_id: int) -> Tuple[List[List[int]], List[int]]:
    batch_src = []
    batch_src_len = []
    for tj in trajectories:
        src, src_len = dqn_input(tj, tokenizer, num_tokens, padding_val_id)
        batch_src.append(src)
        batch_src_len.append(src_len)
    return batch_src, batch_src_len


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
        sep_val_id: int,
        num_tokens: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given one trajectory and its admissible actions, create a training
    set of input for Bert.

    E.g. input: [1, 2, 3], and action_matrix [[1, 3], [2, PAD], [4, PAD]]
    suppose we need length to be 10.
    output:
      [[1, 2, 3, SEP, 1, 3,   SEP, PAD, PAD, PAD],
       [1, 2, 3, SEP, 2, SEP, PAD, PAD, PAD, PAD],
       [1, 2, 3, SEP, 4, SEP, PAD, PAD, PAD, PAD]]
    segment of trajectory and actions:
    [[0, 0, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 1, 1, 0],
     [0, 0, 0, 0, 1, 1, 0]]
    input size:
    [7, 6, 6]
    :param action_matrix:
    :param action_len:
    :param trajectory:
    :param trajectory_len:
    :param sep_val_id:
    :param num_tokens:
    :return: trajectory + action; segmentation ids; sizes
    """
    inp = np.concatenate([
        trajectory[:trajectory_len],
        np.asarray([sep_val_id])])
    n_actions = len(action_matrix)
    action_matrix = np.concatenate(
        [action_matrix, np.zeros([n_actions, 1])], axis=-1)
    action_matrix[
        range(n_actions), action_len] = sep_val_id
    inp = np.repeat(inp[None, :], n_actions, axis=0)
    inp = np.concatenate([inp, action_matrix], axis=-1)
    n_rows, n_cols = inp.shape
    inp = np.concatenate(
        [inp, np.zeros([n_rows, num_tokens - n_cols])], axis=-1)
    inp_size = trajectory_len + action_len + 2
    seg_tj_action = np.zeros_like(inp)
    seg_tj_action[:, trajectory_len + 1:] = 1
    return inp, seg_tj_action, inp_size


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


def get_batch_best_1d_idx_w_mask(
        q_actions: List[np.ndarray],
        mask: np.ndarray = np.asarray([1])) -> List[int]:
    """
    Choose the action idx with the best q value, without choosing from
    inadmissible actions.
    :param q_actions: a batch of q-vectors
    :param mask:
    :return:
    """
    q_actions = np.asarray(q_actions)
    mask = np.ones_like(q_actions) * mask
    inv_mask = np.logical_not(mask)
    min_q_val = np.min(q_actions, axis=-1)
    q_actions = q_actions * mask + min_q_val[:, None] * inv_mask
    action_idx = np.argmax(q_actions, axis=-1)
    return list(action_idx)


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


def get_best_2d_q(q_actions_t, eos_id) -> (list, float):
    """
    </S> also counts for an action, which is the empty action
    the last token should be </S>
    if it's not </S> according to the argmax, then force set it to be </S>.
    Q val for a whole action is the average of all Q val of valid tokens.
    :param q_actions_t: a q-matrix of a state computed from TF at step t
    :param eos_id: end-of-sentence
    """
    action_idx = np.argmax(q_actions_t, axis=1)
    valid_len = 0
    for a in action_idx:
        valid_len += 1
        if a == eos_id:
            break
    padded_action_idx = np.zeros_like(action_idx)
    padded_action_idx[:valid_len] = action_idx[:valid_len]
    # make sure the last token is eos no matter what
    padded_action_idx[valid_len-1] = eos_id
    q_val = np.mean(
        q_actions_t[range(valid_len), padded_action_idx[:valid_len]])
    return padded_action_idx, q_val, valid_len

