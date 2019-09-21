import time
import numpy as np

from deeptextworld.log import Logging
from deeptextworld.utils import flatten


class BaseTrajectory(Logging):
    """
    BaseTrajectory only takes care of interacting with Agent on collecting
    game scripts.
    Fetching data from Trajectory feeding into Encoder should be implemented
    in extended classes.
    """
    def __init__(self):
        super(BaseTrajectory, self).__init__()
        self.trajectories = {}
        self.curr_tj = None
        self.curr_tid = None

    def get_last_sid(self):
        """
        A state is defined as a series of interactions between a game and an
        agent ended with the game's last response.
        e.g. "G0, A1, G2, A3, G4" is a state ended with the game's last response
        named "G4".
        :return:
        """
        return len(self.curr_tj) - 1

    def get_current_tid(self):
        return self.curr_tid

    def request_delete_key(self, key):
        """
        delete all keys with timestamp smaller or equal with key
        """
        keys = sorted(self.trajectories.keys())
        for k in keys:
            if k > key:
                break
            else:
                self.trajectories.pop(k, None)
                self.debug('trajectory {} (time<=) {} deleted'.format(k, key))

    @classmethod
    def _ctime(cls):
        return int(round(time.time() * 1000))

    def add_new_tj(self, tid=None):
        if tid is None or tid in self.trajectories:
            tid = self._ctime()
        if self.curr_tj is not None and len(self.curr_tj) != 0:
            self.trajectories[self.curr_tid] = self.curr_tj
        self.curr_tid = tid
        self.curr_tj = []
        return tid

    def append(self, val):
        self.curr_tj.append(val)

    def save_tjs(self, path):
        tids = list(self.trajectories.keys())
        vals = list(self.trajectories.values())
        np.savez(path, tids=tids, vals=vals,
                 curr_tid=[self.curr_tid], curr_tj=[self.curr_tj])

    def load_tjs(self, path):
        tjs = np.load(path)
        self.curr_tid = tjs["curr_tid"][0]
        self.curr_tj = list(tjs["curr_tj"][0])
        tids = tjs["tids"]
        vals = tjs["vals"]
        assert len(tids) == len(vals), "incompatible trajectory ids and values"
        for i in range(len(tids)):
            self.trajectories[tids[i]] = list(vals[i])

    def fetch_last_state(self):
        raise NotImplementedError()

    def fetch_batch_states(self, b_tid, b_sid):
        raise NotImplementedError()

    def fetch_batch_states_pair(self, b_tid, b_sid):
        raise NotImplementedError()


class StateTextCompanion(BaseTrajectory):

    def add_new_tj(self, tid=None):
        assert tid is not None, "trajectory id must not be None"
        if self.curr_tj is not None and len(self.curr_tj) != 0:
            self.trajectories[self.curr_tid] = self.curr_tj
        self.curr_tid = tid
        self.curr_tj = []
        return tid

    def fetch_last_state(self):
        return self.fetch_raw_state_by_idx(self.curr_tid, self.get_last_sid())

    def fetch_raw_state_by_idx(self, tid, sid):
        if tid == self.curr_tid:  # make sure test cid first
            tj = self.curr_tj
        elif tid in self.trajectories:
            tj = self.trajectories[tid]
        else:
            return None
        state = tj[sid]
        return state, len(state)

    def fetch_batch_states(self, b_tid, b_sid):
        """
        Fetch a batch of states and padding to the same length
        """
        b_states = []
        b_len = []
        for tid, sid in zip(b_tid, b_sid):
            stat, stat_len = self.fetch_raw_state_by_idx(tid, sid)
            b_states.append(stat)
            b_len.append(stat_len)
        return b_states, b_len

    def fetch_batch_states_pair(self, b_tid, b_sid):
        """
        :return: p_states: prior states; s_states: successive states
        """
        b_sid = np.asarray(b_sid)
        p_states, p_len = self.fetch_batch_states(b_tid, b_sid - 1)
        s_states, s_len = self.fetch_batch_states(b_tid, b_sid)
        return p_states, s_states, p_len, s_len


class VarSizeTrajectory(BaseTrajectory):
    """
    VarSizeTrajectory provides variable batch size of sentences for RNN-like
    neural networks.
    batch data size: batch_size * max_sentence_size_in_the_batch
    """
    def __init__(self, hp, padding_val=0):
        super(VarSizeTrajectory, self).__init__()
        self.num_turns = hp.num_turns
        self.padding_val = padding_val

    def _pad_batch_raw_states(self, batch_sentences, lens):
        """
        pad each indexed string to max_len
        """
        filled_lens = list(np.max(lens) - lens)
        padded_batch = ([s_l[0] + [self.padding_val] * s_l[1]
                         for s_l in zip(batch_sentences, filled_lens)])
        return padded_batch, lens

    def fetch_last_state(self):
        return self.fetch_raw_state_by_idx(self.curr_tid, self.get_last_sid())

    def fetch_raw_state_by_idx(self, tid, sid):
        if tid == self.curr_tid:  # make sure test cid first
            tj = self.curr_tj
        elif tid in self.trajectories:
            tj = self.trajectories[tid]
        else:
            return [], 0
        state = flatten(tj[max(0, sid - self.num_turns):sid + 1])
        return state, len(state)

    def fetch_batch_states(self, b_tid, b_sid):
        """
        Fetch a batch of states and padding to the same length
        """
        b_states = []
        b_len = []
        for tid, sid in zip(list(b_tid), list(b_sid)):
            stat, stat_len = self.fetch_raw_state_by_idx(tid, sid)
            b_states.append(stat)
            b_len.append(stat_len)
        b_states, b_len = self._pad_batch_raw_states(b_states, b_len)
        return np.asarray(b_states), np.asarray(b_len)

    def fetch_batch_states_pair(self, b_tid, b_sid):
        """
        :return: p_states: prior states; s_states: successive states
        """
        b_sid = np.asarray(b_sid)
        p_states, p_len = self.fetch_batch_states(b_tid, b_sid - 1)
        s_states, s_len = self.fetch_batch_states(b_tid, b_sid)
        return p_states, s_states, p_len, s_len


class SingleChannelTrajectory(VarSizeTrajectory):
    """
    SingleChannelTrajectory provides fix sentence length one-channel sentences
    for CNN-like neural networks.
    batch data size: batch_size * num_tokens
    """
    def __init__(self, hp, padding_val=0):
        super(SingleChannelTrajectory, self).__init__(hp, padding_val)
        self.num_tokens = hp.num_tokens

    def _pad_raw_state(self, sentence):
        padding_size = self.num_tokens - len(sentence)
        if padding_size >= 0:
            state = sentence + [self.padding_val] * padding_size
        else:
            state = sentence[-padding_size:]
            # self.debug(
            #     "trimming in the front {} tokens".format(-padding_size))
        state_len = min(self.num_tokens, len(sentence))
        return state, state_len

    def fetch_last_state(self):
        state, _ = self.fetch_raw_state_by_idx(
            self.curr_tid, self.get_last_sid())
        return self._pad_raw_state(state)

    def fetch_batch_states(self, b_tid, b_sid):
        b_states = []
        b_len = []
        for tid, sid in zip(b_tid, b_sid):
            stat, _ = self.fetch_raw_state_by_idx(tid, sid)
            padded_state, padded_len = self._pad_raw_state(stat)
            b_states.append(padded_state)
            b_len.append(padded_len)
        return np.asarray(b_states), np.asarray(b_len)


class MultiChannelTrajectory(BaseTrajectory):
    """
    MultiChannelTrajectory provides fix sentence length multi-channel sentences
    for CNN-like neural networks.
    num_turns is treated as multi-channel.
    batch data size: batch_size * num_turns * num_tokens
    """
    def __init__(self, hp, padding_val=0):
        super(MultiChannelTrajectory, self).__init__()
        self.num_turns = hp.num_turns
        self.num_tokens = hp.num_tokens
        self.padding_val = padding_val

    def _pad_raw_state(self, batch_sentences):
        """
        pad a batch of sentences into a state.
        the shape of state is (num_turns, num_tokens)
        e.g. num_turns = 4; num_tokens = 4; pad_val = 0;
        then given sentences (go east; go east OK), it will be padded as
        -------------
        0 0 0 0
        0 0 0 0
        go east 0 0
        go east OK 0
        -------------
        its lengths is
        -------------
        0
        0
        2
        3
        -------------
        sparsity ratio is 5.0 / 16 = 0.31
        """
        batch_lens = np.asarray([len(s) for s in batch_sentences])
        batch_size = len(batch_sentences)
        filled_lens = list(self.num_tokens - batch_lens)
        padded_batch = np.asarray(
            ([s_l[0] +
              [self.padding_val] * s_l[1] if s_l[1] >= 0 else s_l[0][-s_l[1]:]
              for s_l in zip(batch_sentences, filled_lens)]))
        empty_pad = np.full([self.num_turns - batch_size,
                             self.num_tokens],
                            self.padding_val)
        empty_lens = np.zeros(self.num_turns - batch_size)
        state = np.concatenate((empty_pad, padded_batch), axis=0)
        lens = np.concatenate((empty_lens, batch_lens), axis=0)
        sparsity_ratio = (np.sum(batch_lens) * 1. /
                          (self.num_turns * self.num_tokens))
        return state, lens, sparsity_ratio

    def fetch_last_state(self):
        """
        :return: state, lens
        """
        return self._fetch_state_by_idx(self.curr_tid, self.get_last_sid())

    def _fetch_state_by_idx(self, tid, sid):
        """
        :return: state, lens
        """
        if tid == self.curr_tid:  # make sure test cid first
            tj = self.curr_tj
        elif tid in self.trajectories:
            tj = self.trajectories[tid]
        else:
            return None
        raw_state = tj[max(0, sid+1-self.num_turns):sid+1]
        state, lens, sparse_ratio = self._pad_raw_state(raw_state)
        self.debug('sparsity ratio: {}'.format(sparse_ratio))
        return state, lens

    def fetch_batch_states(self, b_tid, b_sid):
        """
        fetch a batch of (state_q, state_q1, lens_q, lens_q1) given
        batch trajectory ids and batch sentence ids.
        returned states in shape (num_batch, num_turns, num_tokens)
        returned lens in shape (num_batch, num_turns)
        """
        b_states = []
        b_len = []
        for tid, sid in zip(b_tid, b_sid):
            stat, stat_len = self._fetch_state_by_idx(tid, sid)
            b_states.append(stat)
            b_len.append(stat_len)
        return np.asarray(b_states), np.asarray(b_len)

    def fetch_batch_states_pair(self, b_tid, b_sid):
        """
        :return: p_states: prior states; s_states: successive states
        """
        b_sid = np.asarray(b_sid)
        p_states, p_len = self.fetch_batch_states(b_tid, b_sid - 1)
        s_states, s_len = self.fetch_batch_states(b_tid, b_sid)
        return p_states, s_states, p_len, s_len
