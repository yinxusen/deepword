from bisect import bisect_left
from multiprocessing.pool import ThreadPool
from os import remove as prm
from typing import Dict, Tuple, List, Any

import numpy as np
from numpy.random import choice as npc

from deepword.agents.base_agent import BaseAgent
from deepword.agents.competition_agent import CompetitionAgent
from deepword.agents.utils import Memolet
from deepword.agents.utils import batch_dqn_input
from deepword.agents.utils import get_path_tags
from deepword.agents.zork_agent import ZorkAgent
from deepword.utils import get_hash


class DSQNAgent(BaseAgent):
    """
    BaseAgent with hs2tj: hash states point to trajectories
    for SNN training
    """
    def __init__(self, hp, model_dir):
        super(DSQNAgent, self).__init__(hp, model_dir)
        self.hs2tj_prefix = "hs2tj"
        self.hash_states2tjs: Dict[str, List[Tuple[int, int]]] = dict()
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
                prm(self._get_context_obj_path_w_tag(self.hs2tj_prefix, tag))

    def _clean_stale_context(self, tids):
        super(DSQNAgent, self)._clean_stale_context(tids)
        if not tids:
            return
        hs2tj_cleaned: Dict[str, List[Tuple[int, int]]] = dict()
        for k in self.hash_states2tjs.keys():
            start_t = bisect_left(
                [t for t, s in self.hash_states2tjs[k]], max(tids))
            if self.hash_states2tjs[k][start_t:]:
                hs2tj_cleaned[k] = self.hash_states2tjs[k][start_t:]
            else:
                self.debug("remove key {} from hs2tj".format(k))
        self.hash_states2tjs = hs2tj_cleaned

    def _collect_new_sample(
            self, master, instant_reward, dones, infos):
        (actions, actions_mask, sys_actions_mask, instant_reward
         ) = super(DSQNAgent, self)._collect_new_sample(
            master, instant_reward, dones, infos)

        if not dones[0]:
            state = self.stc.fetch_last_state()[-1]
            hs = get_hash(state.obs + "\n" + state.inventory)
            if hs not in self.hash_states2tjs:
                self.hash_states2tjs[hs] = []
            last_tid = self.tjs.get_current_tid()
            last_sid = self.tjs.get_last_sid()
            self.hash_states2tjs[hs].append((last_tid, last_sid))
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
        # target key set should contain items more than twice, since we need to
        # separate target set and same set.
        target_key_set = list(
            filter(lambda x: len(self.hash_states2tjs[x]) >= 2,
                   self.hash_states2tjs.keys()))
        self.debug(
            "choose from {} keys for SNN target".format(len(target_key_set)))
        hs_keys = npc(target_key_set, size=batch_size)

        diff_keys_duo = npc(
            list(self.hash_states2tjs.keys()), replace=False,
            size=(batch_size, 2))
        diff_keys = diff_keys_duo[:, 0]
        same_key_ids = np.where(hs_keys == diff_keys)[0]
        diff_keys[same_key_ids] = diff_keys_duo[same_key_ids, 1]

        tgt_set = []
        same_set = []
        diff_set = []
        for hk, dk in zip(hs_keys, diff_keys):
            samples_ids = npc(
                len(self.hash_states2tjs[hk]), size=2, replace=False)
            tgt_set.append(self.hash_states2tjs[hk][samples_ids[0]])
            same_set.append(self.hash_states2tjs[hk][samples_ids[1]])
            diff_set.append(
                self.hash_states2tjs[dk][npc(len(self.hash_states2tjs[dk]))])

        trajectories = [
            self.tjs.fetch_state_by_idx(tid, sid) for tid, sid in
            tgt_set + same_set + diff_set]
        batch_src, batch_src_len, batch_mask = batch_dqn_input(
            trajectories, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id, with_action_padding=False)
        tgt_src = batch_src[: len(tgt_set)]
        tgt_src_len = batch_src_len[: len(tgt_set)]
        same_src = batch_src[len(tgt_set): len(tgt_set) + len(same_set)]
        same_src_len = batch_src_len[len(tgt_set): len(tgt_set) + len(same_set)]
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
