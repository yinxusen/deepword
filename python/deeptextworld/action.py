import time

import numpy as np
from bitarray import bitarray
from bert.tokenization import FullTokenizer

from deeptextworld.log import Logging


class ActionCollector(Logging):
    def __init__(
            self, vocab_file, n_actions=200, n_tokens=10,
            token2idx=None, unk_val_id=None, padding_val_id=None):
        super(ActionCollector, self).__init__()
        # collections of all actions and its indexed vectors
        self.actions_base = {}
        self.action_matrix_base = {}
        self.action_len_base = {}

        # metadata of the action collector
        self.n_actions = n_actions
        self.n_tokens = n_tokens
        self.token2idx = token2idx
        self.unk_val_id = unk_val_id
        self.padding_val_id = padding_val_id

        # current episode actions
        self.action2idx = None
        self.actions = None
        self.curr_aid = 0
        self.curr_eid = None
        self.action_matrix = None
        self.action_len = None

        self.tokenizer = FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

    def init(self):
        self.action2idx = {}
        self.actions = [""] * self.n_actions
        self.curr_aid = 0
        self.curr_eid = None
        self.action_matrix = np.full(
            (self.n_actions, self.n_tokens),
            fill_value=self.padding_val_id, dtype=np.int32)
        self.action_len = np.zeros(self.n_actions, dtype=np.int32)

    @classmethod
    def _ctime(cls):
        return int(round(time.time() * 1000))

    def add_new_episode(self, eid=None):
        if eid is None:
            eid = self._ctime()

        if eid == self.curr_eid:
            # self.info("continue current episode: {}".format(eid))
            return

        # self.info("add new episode in actions: {}".format(eid))

        if self.size() != 0 and self.curr_eid is not None:
            self.actions_base[self.curr_eid] = self.actions[:self.size()]
            self.action_matrix_base[self.curr_eid] = self.action_matrix
            self.action_len_base[self.curr_eid] = self.action_len

        self.init()
        self.curr_eid = eid
        if self.curr_eid in self.actions_base:
            self.info("found existing episode: {}".format(self.curr_eid))
            self.curr_aid = len(self.actions_base[self.curr_eid])
            self.actions[:self.size()] = self.actions_base[self.curr_eid]
            self.action_matrix = self.action_matrix_base[self.curr_eid]
            self.action_len = self.action_len_base[self.curr_eid]
            self.action2idx = dict([(a, i) for (i, a) in
                                    enumerate(self.actions)])
            self.info("{} actions loaded".format(self.size()))
        else:
            pass

    def _preprocess_action(self, action):
        return self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(action))

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
            if a not in self.action2idx:
                assert self.curr_aid < self.n_actions - 1, "n_actions too small"
                self.action2idx[a] = self.curr_aid
                action_idx = self._preprocess_action(a)
                n_action_tokens = len(action_idx)
                if n_action_tokens > self.n_tokens:
                    self.warning("trimming action {} size {} -> {}".format(
                        a, n_action_tokens, self.n_tokens))
                    n_action_tokens = self.n_tokens
                self.action_len[self.curr_aid] = n_action_tokens
                self.action_matrix[self.curr_aid][:n_action_tokens] =\
                    action_idx[:n_action_tokens]
                self.actions[self.curr_aid] = a
                # self.debug("learn new admissible command: {}".format(a))
                self.curr_aid += 1
            bit_mask_vec[self.action2idx[a]] = True
        return bit_mask_vec.tobytes()

    def get_action_matrix(self, eid=None):
        if eid is None or eid == self.curr_eid:
            return self.action_matrix
        else:
            return self.action_matrix_base[eid]

    def get_action_len(self, eid=None):
        if eid is None or eid == self.curr_eid:
            return self.action_len
        else:
            return self.action_len_base[eid]

    def get_actions(self, eid=None):
        if eid is None or eid == self.curr_eid:
            return self.actions
        else:
            return self.actions_base[eid]

    def get_action2idx(self, eid=None):
        if eid is None or eid == self.curr_eid:
            return self.action2idx
        else:
            action2idx = dict(
                [(a, i) for (i, a) in enumerate(self.actions_base[eid])])
            return action2idx

    def size(self):
        return self.curr_aid

    def save_actions(self, path):
        metadata = ([self.n_actions, self.n_tokens, self.unk_val_id,
                     self.padding_val_id])
        if self.size() != 0 and self.curr_eid is not None:
            self.actions_base[self.curr_eid] = self.actions[:self.size()]
        actions_base_keys = list(self.actions_base.keys())
        actions_base_vals = list(self.actions_base.values())
        np.savez(path, metadata=metadata,
                 actions_base_keys=actions_base_keys,
                 actions_base_vals=actions_base_vals,
                 action_matrix=[self.action_matrix_base])

    def load_actions(self, path):
        saved = np.load(path)
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
