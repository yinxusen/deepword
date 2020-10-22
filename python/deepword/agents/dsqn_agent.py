from multiprocessing.pool import ThreadPool
from os import remove
from typing import Dict, Tuple, List, Any

import numpy as np

from deepword.agents.base_agent import BaseAgent
from deepword.agents.competition_agent import CompetitionAgent
from deepword.agents.utils import Memolet
from deepword.agents.utils import get_path_tags
from deepword.agents.utils import get_snn_keys
from deepword.agents.zork_agent import ZorkAgent


class DSQNAgent(BaseAgent):
    """
    BaseAgent with hs2tj: hash states point to trajectories
    for SNN training
    """
    def __init__(self, hp, model_dir):
        super(DSQNAgent, self).__init__(hp, model_dir)
        self.hs2tj_prefix = "hs2tj"
        self.hash_states2tjs: Dict[str, Dict[int, List[int]]] = dict()
        self.pool_train = ThreadPool(processes=2)

    def _init_hs2tj(
            self, hs2tj_path: str,
            with_loading: bool = True) -> Dict[str, List[Tuple[int, int]]]:
        hash_states2tjs = dict()
        if with_loading:
            try:
                hs2tj = np.load(hs2tj_path, allow_pickle=True)
                hash_states2tjs = hs2tj["hs2tj"][0]
                self.debug("load hash_states2tjs from file")
            except IOError as e:
                self.debug("load hash_states2tjs error:\n{}".format(e))
        return hash_states2tjs

    def _load_context_objs(self):
        # load others
        super(DSQNAgent, self)._load_context_objs()
        # load hs2tj
        hs2tj_path = self._get_context_obj_path(self.hs2tj_prefix)
        self.hash_states2tjs = self._init_hs2tj(
            hs2tj_path, with_loading=self.is_training)

    def _save_context_objs(self):
        super(DSQNAgent, self)._save_context_objs()
        hs2tj_path = self._get_context_obj_new_path(self.hs2tj_prefix)
        np.savez(hs2tj_path, hs2tj=[self.hash_states2tjs])

    def _get_compatible_snapshot_tag(self):
        # get parent valid tags
        valid_tags = super(DSQNAgent, self)._get_compatible_snapshot_tag()
        valid_tags = set(valid_tags)
        # mix valid tags w/ context objs
        hs2tj_tags = get_path_tags(self.model_dir, self.hs2tj_prefix)
        valid_tags.intersection_update(hs2tj_tags)
        return list(valid_tags)

    def _delete_stale_context_objs(self):
        super(DSQNAgent, self)._delete_stale_context_objs()
        if self._stale_tags is not None:
            for tag in self._stale_tags:
                remove(self._get_context_obj_path_w_tag(self.hs2tj_prefix, tag))

    def _clean_stale_context(self, tids):
        """
        We don't call super method, since we need to know trashed elements
        for removing stale hash_state2tjs.
        """
        self.tjs.request_delete_keys(tids)
        trashed = self.stc.request_delete_keys(tids)
        inverse_trashed = {}
        for tid in trashed:
            for state in trashed[tid]:
                if state.hs not in inverse_trashed:
                    inverse_trashed[state.hs] = []
                inverse_trashed[state.hs].append(tid)
        self.debug("to trash: {}".format(inverse_trashed))
        for hs in inverse_trashed:
            for tid in inverse_trashed[hs]:
                if tid in self.hash_states2tjs[hs]:
                    self.hash_states2tjs[hs].pop(tid)
            if not self.hash_states2tjs[hs]:
                self.hash_states2tjs.pop(hs)

    def _collect_new_sample(
            self, master, instant_reward, dones, infos):
        (actions, actions_mask, sys_actions_mask, instant_reward
         ) = super(DSQNAgent, self)._collect_new_sample(
            master, instant_reward, dones, infos)

        if not dones[0]:
            state = self.stc.fetch_last_state()[-1]
            hs = state.hs
            if hs not in self.hash_states2tjs:
                self.hash_states2tjs[hs] = dict()
            last_tid = self.tjs.get_current_tid()
            last_sid = self.tjs.get_last_sid()
            if last_tid not in self.hash_states2tjs[hs]:
                self.hash_states2tjs[hs][last_tid] = []
            self.hash_states2tjs[hs][last_tid].append(last_sid)
        else:
            pass  # final states are not considered

        return actions, actions_mask, sys_actions_mask, instant_reward

    def get_snn_pairs(
            self, batch_size: int) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample SNN pairs for SNN part training

        Args:
            batch_size: how many data points to generate. Notice that
             batch_size * 2 data points will be generated, one half for
             trajectory pairs with the same states; the other half for
             trajectory pairs with different states.

        Returns:
            src: trajectories
            src_len: length of them
            src2: the paired trajectories
            src2_len: length of them
            labels: `0` for same states; `1` for different states
        """
        target_set, same_set, diff_set = get_snn_keys(
            self.hash_states2tjs, self.tjs, batch_size)

        trajectories = [
            self.tjs.fetch_state_by_idx(tid, sid) for tid, sid in
            target_set + same_set + diff_set]
        batch_src, batch_src_len = self.core.batch_trajectory2input(
            trajectories)
        tgt_src = batch_src[: len(target_set)]
        tgt_src_len = batch_src_len[: len(target_set)]
        same_src = batch_src[len(target_set): len(target_set) + len(same_set)]
        same_src_len = batch_src_len[
            len(target_set): len(target_set) + len(same_set)]
        diff_src = batch_src[-len(diff_set):]
        diff_src_len = batch_src_len[-len(diff_set):]

        src = np.concatenate([tgt_src, tgt_src], axis=0)
        src_len = np.concatenate([tgt_src_len, tgt_src_len], axis=0)
        src2 = np.concatenate([same_src, diff_src], axis=0)
        src2_len = np.concatenate([same_src_len, diff_src_len], axis=0)
        labels = np.concatenate(
            [np.zeros(batch_size), np.ones(batch_size)], axis=0)
        return src, src_len, src2, src2_len, labels

    def save_train_pairs(
            self, t: int, src: np.ndarray, src_len: np.ndarray,
            src2: np.ndarray, src2_len: np.ndarray, labels: np.ndarray) -> None:
        """
        Save SNN pairs for verification.

        Args:
            t: current training steps
            src: trajectories
            src_len: length of trajectories
            src2: paired trajectories
            src2_len: length of paired trajectories
            labels: `0` or `1` for same or different states
        """
        src_str = []
        for s in src:
            src_str.append(" ".join(
                map(lambda i: self.tokenizer.inv_vocab[i],
                    filter(lambda x: x != self.hp.padding_val_id, s))
            ))
        src2_str = []
        for s in src2:
            src2_str.append(" ".join(
                map(lambda i: self.tokenizer.inv_vocab[i],
                    filter(lambda x: x != self.hp.padding_val_id, s))
            ))
        np.savez(
            "{}/{}-{}.npz".format(self.model_dir, "train-pairs", t),
            src=src_str, src2=src2_str, src_len=src_len, src2_len=src2_len,
            labels=labels)

    def _prepare_other_train_data(self, b_memory: List[Memolet]) -> Any:
        async_snn_data = self.pool_train.apply_async(
            self.get_snn_pairs, args=(self.hp.batch_size,))
        return async_snn_data


class DSQNCompetitionAgent(DSQNAgent, CompetitionAgent):
    # TODO: Multi-inheritance is dangerous.
    #     Make sure there are no overlapped method overriding for both parents.
    pass


class DSQNZorkAgent(DSQNAgent, ZorkAgent):
    # TODO: Multi-inheritance is dangerous.
    #     Make sure there are no overlapped method overriding for both parents.
    pass
