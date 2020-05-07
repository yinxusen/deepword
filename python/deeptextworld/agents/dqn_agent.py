from copy import deepcopy
from os.path import join as pjoin
from typing import Dict, Optional, List, Any

import numpy as np

from deeptextworld.agents.base_agent import BaseCore, BaseAgent
from deeptextworld.agents.base_agent import TFCore, ActionDesc, ACT_TYPE
from deeptextworld.agents.utils import ActionMaster
from deeptextworld.agents.utils import ObsInventory
from deeptextworld.agents.utils import get_best_1d_q
from deeptextworld.utils import get_hash


class DQNCore(TFCore):
    """
    DQNAgent that treats actions as types
    """

    def __init__(self, hp, model_dir, tokenizer):
        super(DQNCore, self).__init__(hp, model_dir, tokenizer)

    def get_a_policy_action(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            actions: List[str],
            action_mask: np.ndarray,
            cnt_action: Optional[Dict[int, float]]) -> ActionDesc:
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        """
        src, src_len, _ = self.trajectory2input(trajectory)
        q_actions = self.sess.run(self.model.q_actions, feed_dict={
            self.model.src_: [src],
            self.model.src_len_: [src_len]
        })[0]

        cnt_action_array = []
        for mid in action_mask:
            cnt_action_array.append(
                cnt_action[mid] if mid in cnt_action else 0.)

        admissible_q_actions = q_actions[action_mask]
        action_idx, q_val = get_best_1d_q(
            admissible_q_actions - cnt_action_array)
        real_action_idx = action_mask[action_idx]
        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_drrn,
            action_idx=real_action_idx,
            token_idx=action_matrix[real_action_idx],
            action_len=action_len[real_action_idx],
            action=actions[real_action_idx],
            q_actions=admissible_q_actions)
        return action_desc

    def _compute_expected_q(
            self,
            action_mask: List[np.ndarray],
            trajectories: List[List[ActionMaster]],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:
        """
        Compute expected q values given post trajectories and post actions

        notice that action_mask, tids, sids should belong to post game states,
        while dones, rewards belong to pre game states.
        """

        src, src_len, _ = self.batch_trajectory2input(trajectories)
        target_model, target_sess = self.get_target_model()
        # target network provides the value used as expected q-values
        qs_target = target_sess.run(
            target_model.q_actions,
            feed_dict={
                target_model.src_: src,
                target_model.src_len_: src_len})

        # current network decides which action provides best q-value
        qs_dqn = self.sess.run(
            self.model.q_actions,
            feed_dict={
                self.model.src_: src,
                self.model.src_len_: src_len})

        expected_q = np.zeros_like(rewards)
        for i in range(len(expected_q)):
            expected_q[i] = rewards[i]
            if not dones[i]:
                action_idx, _ = get_best_1d_q(qs_dqn[i, action_mask[i]])
                real_action_idx = action_mask[i][action_idx]
                expected_q[i] += (
                        self.hp.gamma * qs_target[i, real_action_idx])
        return expected_q

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int, others: Any) -> np.ndarray:

        expected_q = self._compute_expected_q(
            action_mask=post_action_mask, trajectories=post_trajectories,
            dones=dones, rewards=rewards)

        pre_src, pre_src_len, _ = self.batch_trajectory2input(pre_trajectories)
        _, summaries, loss_eval, abs_loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={
                self.model.src_: pre_src,
                self.model.src_len_: pre_src_len,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: action_idx,
                self.model.expected_q_: expected_q})

        self.info('loss: {}'.format(loss_eval))
        self.train_summary_writer.add_summary(
            summaries, step - self.hp.observation_t)
        return abs_loss


class TabularCore(BaseCore):
    """
    Tabular-wise DQN agent that uses matrix to store q-vectors and uses
    hashed values of observation + inventory as game states
    """
    def __init__(self, hp, model_dir, tokenizer):
        super(TabularCore, self).__init__(hp, model_dir, tokenizer)
        self.hp = hp
        self.q_mat_prefix = "q_mat"
        # model of tabular Q-learning, map from state to q-vectors
        self.q_mat: Dict[str, np.ndarray] = dict()
        self.target_q_mat: Dict[str, np.ndarray] = dict()
        self.state2hash: Dict[ObsInventory, str] = dict()
        self.tokenizer = tokenizer
        self.model_dir = model_dir
        self.ckpt_prefix = "after-epoch"
        self.ckpt_path = pjoin(self.model_dir, self.q_mat_prefix)

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int, others: Any) -> np.ndarray:

        expected_q = self._compute_expected_q(
            post_action_mask, post_states, dones, rewards)

        pre_hash_states = [
            self.get_state_hash(state[0]) for state in pre_states]

        abs_loss = np.zeros_like(rewards)
        for i, ps in enumerate(pre_hash_states):
            if ps not in self.q_mat:
                self.q_mat[ps] = np.zeros(self.hp.n_actions)
            prev_q_val = self.q_mat[ps][action_idx[i]]
            delta_q_val = expected_q[i] - prev_q_val
            abs_loss[i] = abs(delta_q_val)
            self.q_mat[ps][action_idx[i]] = (
                    prev_q_val + delta_q_val * b_weight[i])
        return abs_loss

    def get_a_policy_action(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            actions: List[str],
            action_mask: np.ndarray,
            cnt_action: Optional[Dict[int, float]]) -> ActionDesc:

        hs = self.get_state_hash(state)
        q_actions = self.q_mat.get(hs, np.zeros(self.hp.n_actions))
        admissible_q_actions = q_actions[action_mask]
        action_idx, q_val = get_best_1d_q(admissible_q_actions)
        real_action_idx = action_mask[action_idx]
        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_tbl,
            action_idx=real_action_idx,
            token_idx=action_matrix[real_action_idx],
            action_len=action_len[real_action_idx],
            action=actions[real_action_idx])
        return action_desc

    def create_or_reload_target_model(
            self, restore_from: Optional[str] = None) -> None:
        self.target_q_mat = deepcopy(self.q_mat)

    def init(
            self, is_training: bool, load_best: bool = False,
            restore_from: Optional[str] = None) -> None:
        self.is_training = is_training
        try:
            if not restore_from:
                tags = BaseAgent.get_path_tags(
                    self.ckpt_path, self.ckpt_prefix)
                self.loaded_ckpt_step = max(tags)
                restore_from = pjoin(
                    self.ckpt_path, "{}-{}.npz".format(
                        self.ckpt_prefix, self.loaded_ckpt_step))
            else:
                # TODO: fetch loaded ckpt step
                pass
            npz_q_mat = np.load(restore_from, allow_pickle=True)
            q_mat_key = npz_q_mat["q_mat_key"]
            q_mat_val = npz_q_mat["q_mat_val"]
            self.q_mat = dict(zip(q_mat_key, q_mat_val))
            self.debug("load q_mat from file")
            self.target_q_mat = deepcopy(self.q_mat)
            self.debug("init target_q_mat with q_mat")
        except IOError as e:
            self.debug("load q_mat error:\n{}".format(e))
        pass

    def save_model(self, t: Optional[int] = None) -> None:
        q_mat_path = pjoin(
            self.ckpt_path, "{}-{}.npz".format(self.ckpt_prefix, t))
        np.savez(
            q_mat_path,
            q_mat_key=list(self.q_mat.keys()),
            q_mat_val=list(self.q_mat.values()))

    def get_state_hash(self, state: ObsInventory) -> str:
        if state in self.state2hash:
            hs = self.state2hash[state]
        else:
            hs = get_hash(state.obs + "\n" + state.inventory)
            self.state2hash[state] = hs
        return hs

    def _compute_expected_q(
            self,
            action_mask: List[np.ndarray],
            states: List[ObsInventory],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:
        post_hash_states = [self.get_state_hash(state) for state in states]
        post_qs_target = np.asarray(
            [self.target_q_mat.get(s, np.zeros(self.hp.n_actions))
             for s in post_hash_states])
        post_qs_dqn = np.asarray(
            [self.q_mat.get(s, np.zeros(self.hp.n_actions))
             for s in post_hash_states])

        expected_q = np.zeros_like(rewards)
        for i in range(len(expected_q)):
            expected_q[i] = rewards[i]
            if not dones[i]:
                action_idx, _ = get_best_1d_q(post_qs_dqn[i, action_mask[i]])
                real_action_idx = action_mask[i][action_idx]
                expected_q[i] += (
                        self.hp.gamma * post_qs_target[i, real_action_idx])
        return expected_q
