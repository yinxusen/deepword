from collections import namedtuple

from nltk import word_tokenize

from deeptextworld.log import Logging
from deeptextworld.utils import load_vocab, get_token2idx


class DRRNMemo(namedtuple(
    "DRRNMemo",
    ("tid", "sid", "gid", "aid", "token_id", "a_len", "reward", "is_terminal",
     "action_mask", "next_action_mask"))):
    pass


class DRRNMemoTeacher(namedtuple(
    "DRRNMemoTeacher",
    ("tid", "sid", "gid", "aid", "reward", "is_terminal",
     "action_mask", "next_action_mask", "q_actions"))):
    pass


class ActionDesc(namedtuple(
    "ActionDesc",
    ("action_type", "action_idx", "token_idx",
     "action_len", "action"))):
    def __repr__(self):
        return "{}/{}/{}/{}/{}".format(
            self.action_type, self.action_idx, self.token_idx, self.action_len,
            self.action)


class NLTKTokenizer(object):
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
    ("recipe", "desc", "inventory", "max_score", "won",
     "actions", "templates", "verbs", "entities"))):
    pass


INFO_KEY = EnvInfosKey(
    recipe="extra.recipe",
    desc="description",
    inventory="inventory",
    max_score="max_score",
    won="won",
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
        self.debug("eps: {}".format(eps_t))
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

