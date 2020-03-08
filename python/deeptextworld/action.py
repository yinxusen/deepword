import time

import numpy as np
from bitarray import bitarray

from deeptextworld.log import Logging


class ActionCollector(Logging):
    def __init__(
            self, tokenizer, n_actions=200, n_tokens=10,
            unk_val_id=None, padding_val_id=None, eos_id=None, pad_eos=False):
        super(ActionCollector, self).__init__()
        # collections of all actions and its indexed vectors
        self.actions_base = {}
        self.action_matrix_base = {}
        self.action_len_base = {}

        # metadata of the action collector
        self.n_actions = n_actions
        self.n_tokens = n_tokens
        self.unk_val_id = unk_val_id
        self.padding_val_id = padding_val_id
        self.eos_id = eos_id

        # current episode actions
        self._action2idx = None
        self._actions = None
        self._curr_aid = 0
        self._curr_eid = None
        self._action_matrix = None
        self._action_len = None

        self.tokenizer = tokenizer
        self.pad_eos = pad_eos

    def init(self):
        self._action2idx = {}
        self._actions = [""] * self.n_actions
        self._curr_aid = 0
        self._curr_eid = None
        self._action_matrix = np.full(
            (self.n_actions, self.n_tokens),
            fill_value=self.padding_val_id, dtype=np.int32)
        self._action_len = np.zeros(self.n_actions, dtype=np.int32)

    @classmethod
    def _ctime(cls):
        return int(round(time.time() * 1000))

    def add_new_episode(self, eid=None):
        if eid is None:
            eid = self._ctime()

        if eid == self._curr_eid:
            # self.info("continue current episode: {}".format(eid))
            return

        # self.info("add new episode in actions: {}".format(eid))

        if self.size() != 0 and self._curr_eid is not None:
            self.actions_base[self._curr_eid] = self._actions[:self.size()]
            self.action_matrix_base[self._curr_eid] = self._action_matrix
            self.action_len_base[self._curr_eid] = self._action_len

        self.init()
        self._curr_eid = eid
        if self._curr_eid in self.actions_base:
            self.info("found existing episode: {}".format(self._curr_eid))
            self._curr_aid = len(self.actions_base[self._curr_eid])
            self._actions[:self.size()] = self.actions_base[self._curr_eid]
            self._action_matrix = self.action_matrix_base[self._curr_eid]
            self._action_len = self.action_len_base[self._curr_eid]
            self._action2idx = dict([(a, i) for (i, a) in
                                     enumerate(self._actions)])
            self.info("{} actions loaded".format(self.size()))
        else:
            pass

    def idx_tokens(self, action):
        action_idx = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(action))
        action_idx = action_idx[:min(self.n_tokens, len(action_idx))]
        n_action_tokens = len(action_idx)
        if self.pad_eos:
            if n_action_tokens == self.n_tokens:
                action_idx[-1] = self.eos_id
            else:
                action_idx.append(self.eos_id)
                n_action_tokens = len(action_idx)
        else:
            pass
        return action_idx, n_action_tokens

    def extend(self, actions):
        """
        Extend actions into ActionCollector.
        Fail if #actions larger than hp.n_actions - 1.
        :param actions:
        :return:
        """
        bit_mask_vec = bitarray(self.n_actions, endian="little")
        bit_mask_vec[::] = False
        bit_mask_vec[-1] = True  # to avoid tail trimming for bytes
        for a in actions:
            if a not in self._action2idx:
                assert self._curr_aid < self.n_actions - 1, \
                    "n_actions too small"
                self._action2idx[a] = self._curr_aid
                action_idx, n_action_tokens = self.idx_tokens(a)
                self._action_len[self._curr_aid] = n_action_tokens
                self._action_matrix[self._curr_aid][:n_action_tokens] =\
                    action_idx[:n_action_tokens]
                self._actions[self._curr_aid] = a
                self._curr_aid += 1
            bit_mask_vec[self._action2idx[a]] = True
        return bit_mask_vec.tobytes()

    def get_action_matrix(self, eid=None):
        if eid is None or eid == self._curr_eid:
            return self._action_matrix
        else:
            return self.action_matrix_base[eid]

    @property
    def action_matrix(self):
        return self._action_matrix

    def get_action_len(self, eid=None):
        if eid is None or eid == self._curr_eid:
            return self._action_len
        else:
            return self.action_len_base[eid]

    @property
    def action_len(self):
        return self._action_len

    def get_actions(self, eid=None):
        if eid is None or eid == self._curr_eid:
            return self._actions
        else:
            return self.actions_base[eid]

    @property
    def actions(self):
        return self._actions

    def get_action2idx(self, eid=None):
        if eid is None or eid == self._curr_eid:
            return self._action2idx
        else:
            action2idx = dict(
                [(a, i) for (i, a) in enumerate(self.actions_base[eid])])
            return action2idx

    @property
    def action2idx(self):
        return self._action2idx

    def size(self):
        return self._curr_aid

    def save_actions(self, path):
        metadata = ([self.n_actions, self.n_tokens, self.unk_val_id,
                     self.padding_val_id])
        if self.size() != 0 and self._curr_eid is not None:
            self.actions_base[self._curr_eid] = self._actions[:self.size()]
        actions_base_keys = list(self.actions_base.keys())
        actions_base_vals = list(self.actions_base.values())
        np.savez(path, metadata=metadata,
                 actions_base_keys=actions_base_keys,
                 actions_base_vals=actions_base_vals,
                 action_matrix=[self.action_matrix_base])

    def load_actions(self, path):
        saved = np.load(path, allow_pickle=True)
        metadata = saved["metadata"]
        assert len(metadata) == 4, "wrong saved actions format"
        (n_actions, n_tokens, unk_val_id, padding_val_id) = tuple(metadata)
        if self.n_actions < n_actions:
            self.warning("new/loaded #actions: {}/{}".format(
                self.n_actions, n_actions))
        if self.n_tokens < n_tokens:
            self.warning("new/loaded #tokens: {}/{}".format(
                self.n_tokens, n_tokens))
        if self.unk_val_id != unk_val_id:
            self.warning("new/loaded unknown val id: {}/{}".format(
                self.unk_val_id, unk_val_id))
        if self.padding_val_id != padding_val_id:
            self.warning("new/loaded padding val id: {}/{}".format(
                self.padding_val_id, padding_val_id))

        actions_base = dict(zip(list(saved["actions_base_keys"]),
                                list(saved["actions_base_vals"])))
        for eid in actions_base:
            self.add_new_episode(eid)
            self.extend(actions_base[eid])
