from bisect import bisect_left
from multiprocessing.pool import ThreadPool
from os import remove as prm
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
from numpy.random import choice as npc

from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.agents.drrn_agent import DRRNCore
from deeptextworld.agents.utils import ActionMaster, ObsInventory
from deeptextworld.agents.utils import batch_drrn_action_input
from deeptextworld.agents.utils import convert_real_id_to_group_id
from deeptextworld.models.export_models import DSQNModel


class DSQNCore(DRRNCore):
    def __init__(self, hp, model_dir, tokenizer):
        super(DRRNCore, self).__init__(hp, model_dir, tokenizer)
        self.model: Optional[DSQNModel] = None
        self.target_model: Optional[DSQNModel] = None

    def train_one_batch(
            self, pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: np.ndarray,
            post_action_mask: np.ndarray, dones: List[bool],
            rewards: List[float], action_idx: List[int],
            b_weight: np.ndarray, step: int, others: Any) -> np.ndarray:

        expected_q = self._compute_expected_q(
            post_action_mask, post_trajectories, action_matrix, action_len,
            dones, rewards)

        src, src_len, src2, src2_len, labels = others.get()
        pre_src, pre_src_len, _ = self.batch_trajectory2input(pre_trajectories)
        (actions, actions_lens, actions_repeats, group_inv_valid_idx
         ) = batch_drrn_action_input(
            action_matrix, action_len, pre_action_mask)
        group_action_id = convert_real_id_to_group_id(
            action_idx, group_inv_valid_idx, actions_repeats)

        _, summaries, weighted_loss, abs_loss = self.sess.run(
            [self.model.merged_train_op, self.model.weighted_train_summary_op,
             self.model.weighted_loss, self.model.abs_loss],
            feed_dict={
                self.model.src_: pre_src,
                self.model.src_len_: pre_src_len,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: group_action_id,
                self.model.actions_mask_: pre_action_mask,
                self.model.expected_q_: expected_q,
                self.model.actions_: actions,
                self.model.actions_len_: actions_lens,
                self.model.actions_repeats_: actions_repeats,
                self.model.snn_src_: src,
                self.model.snn_src_len_: src_len,
                self.model.snn_src2_: src2,
                self.model.snn_src2_len_: src2_len,
                self.model.labels_: labels})

        return abs_loss

    # TODO: refine the code
    def eval_snn(
            self,
            snn_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray],
            batch_size: int = 32) -> float:

        src, src_len, src2, src2_len, labels = snn_data
        eval_data_size = len(src)
        self.info("start eval with size {}".format(eval_data_size))
        n_iter = (eval_data_size // batch_size) + 1
        total_acc = 0
        total_samples = 0
        for i in range(n_iter):
            self.debug("eval snn iter {} total {}".format(i, n_iter))
            non_empty_src = list(filter(
                lambda x: x[1][0] != 0 and x[1][1] != 0,
                enumerate(zip(src_len, src2_len))))
            non_empty_src_idx = [x[0] for x in non_empty_src]
            src = src[non_empty_src_idx, :]
            src_len = src_len[non_empty_src_idx]
            src2 = src2[non_empty_src_idx, :]
            src2_len = src2_len[non_empty_src_idx]
            labels = labels[non_empty_src_idx]
            labels = labels.astype(np.int32)
            pred, diff_two_states = self.sess.run(
                [self.model.pred, self.model.diff_two_states],
                feed_dict={self.model.snn_src_: src,
                           self.model.snn_src2_: src2,
                           self.model.snn_src_len_: src_len,
                           self.model.snn_src2_len_: src2_len})
            pred_labels = (pred > 0).astype(np.int32)
            total_acc += np.sum(np.equal(labels, pred_labels))
            total_samples += len(src)
        if total_samples == 0:
            avg_acc = -1
        else:
            avg_acc = total_acc * 1. / total_samples
            self.debug("valid sample size {}".format(total_samples))
        return avg_acc
    pass


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

    def init_hs2tj(
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
        self.hash_states2tjs = self.init_hs2tj(
            hs2tj_path, with_loading=self.is_training)

    def _save_context_objs(self):
        super(DSQNAgent, self)._save_context_objs()
        hs2tj_path = self._get_context_obj_new_path(self.hs2tj_prefix)
        np.savez(hs2tj_path, hs2tj=[self.hash_states2tjs])

    def get_compatible_snapshot_tag(self):
        # get parent valid tags
        valid_tags = super(DSQNAgent, self).get_compatible_snapshot_tag()
        valid_tags = set(valid_tags)
        # mix valid tags w/ context objs
        hs2tj_tags = self.get_path_tags(self.model_dir, self.hs2tj_prefix)
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
            if not self.hash_states2tjs[k][start_t:]:
                hs2tj_cleaned[k] = self.hash_states2tjs[k][start_t:]
            else:
                self.debug("remove key {} from hs2tj".format(k))
        self.hash_states2tjs = hs2tj_cleaned

    def collect_new_sample(
            self, master, instant_reward, dones, infos):
        (actions, all_actions, actions_mask, sys_actions_mask, instant_reward
         ) = super(DSQNAgent, self).collect_new_sample(
            master, instant_reward, dones, infos)

        if not dones[0]:
            hs, _ = self.stc.fetch_last_state()
            if hs not in self.hash_states2tjs:
                self.hash_states2tjs[hs] = []
            last_tid = self.tjs.get_current_tid()
            last_sid = self.tjs.get_last_sid()
            self.hash_states2tjs[hs].append((last_tid, last_sid))
        else:
            pass  # final states are not considered

        return (
            actions, all_actions, actions_mask, sys_actions_mask,
            instant_reward)

    def get_snn_pairs(
            self, batch_size: int) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        tgt_src, tgt_src_len = self.tjs.fetch_batch_states_impl(tgt_set)
        same_src, same_src_len = self.tjs.fetch_batch_states_impl(same_set)
        diff_src, diff_src_len = self.tjs.fetch_batch_states_impl(diff_set)

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

    def train_one_batch(self) -> None:
        """
        Train one batch of samples.
        Load target model if not exist, save current model when necessary.
        """
        # if there is not a well-trained model, it is unreasonable
        # to use target model.

        async_snn_data = self.pool_train.apply_async(
            self.get_snn_pairs, args=(self.hp.batch_size,))

        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)

        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        action_id = [m.aid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        reward = [m.reward for m in b_memory]
        is_terminal = [m.is_terminal for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        next_action_mask = [m.next_action_mask for m in b_memory]

        pre_action_mask = self.from_bytes(action_mask)
        post_action_mask = self.from_bytes(next_action_mask)

        post_trajectories = self.tjs.fetch_batch_states(trajectory_id, state_id)
        pre_trajectories = self.tjs.fetch_batch_states(
            trajectory_id, [sid - 1 for sid in state_id])

        post_states = [
            state[0] for state in
            self.stc.fetch_batch_states(trajectory_id, state_id)]
        pre_states = [
            state[0] for state in self.stc.fetch_batch_states(
                trajectory_id, [sid - 1 for sid in state_id])]

        # make sure the p_states and s_states are in the same game.
        # otherwise, it won't make sense to use the same action matrix.
        action_len = (
            [self.actor.get_action_len(gid) for gid in game_id])
        max_action_len = np.max(action_len)
        action_matrix = (
            [self.actor.get_action_matrix(gid)[:, :max_action_len]
             for gid in game_id])

        b_weight = self.core.train_one_batch(
            pre_trajectories=pre_trajectories,
            post_trajectories=post_trajectories,
            pre_states=pre_states,
            post_states=post_states,
            action_matrix=action_matrix,
            action_len=action_len,
            pre_action_mask=pre_action_mask,
            post_action_mask=post_action_mask,
            dones=is_terminal,
            rewards=reward,
            action_idx=action_id,
            b_weight=b_weight,
            step=self.total_t,
            others=async_snn_data)

        self.memo.batch_update(b_idx, b_weight)

        if self.is_time_to_save():
            self.save_snapshot()
            self.core.save_model()
            self.core.create_or_reload_target_model()
