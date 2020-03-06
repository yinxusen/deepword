from typing import Optional, List, Any

import numpy as np

from deeptextworld.agents.base_agent import TFCore, ActionDesc, ACT_TYPE
from deeptextworld.agents.utils import dqn_input, \
    batch_dqn_input, batch_drrn_action_input, get_batch_best_1d_idx, \
    convert_real_id_to_group_id, ActionMaster, ObsInventory
from deeptextworld.agents.utils import get_best_1d_q
from deeptextworld.models.export_models import DRRNModel


class DRRNCore(TFCore):
    """
    DRRN agent that treats actions as meaningful sentences
    """

    def __init__(self, hp, model_dir, tokenizer):
        super(DRRNCore, self).__init__(hp, model_dir, tokenizer)
        self.model: Optional[DRRNModel] = None
        self.target_model: Optional[DRRNModel] = None

    def get_a_policy_action(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray, actions: List[str],
            action_mask: np.ndarray) -> ActionDesc:
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        """
        mask_idx = np.where(action_mask == 1)[0]
        admissible_action_matrix = action_matrix[mask_idx, :]
        admissible_action_len = action_len[mask_idx]

        src, src_len = dqn_input(
            trajectory, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id)
        q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
            self.model.src_: [src],
            self.model.src_len_: [src_len],
            self.model.actions_: admissible_action_matrix,
            self.model.actions_len_: admissible_action_len
        })[0]
        action_idx, q_val = get_best_1d_q(
            q_actions_t - self._cnt_action[mask_idx])
        real_action_idx = mask_idx[action_idx]
        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_drrn,
            action_idx=real_action_idx,
            token_idx=action_matrix[real_action_idx],
            action_len=action_len[real_action_idx],
            action=actions[real_action_idx])
        return action_desc

    def _compute_expected_q(
            self,
            action_mask: np.ndarray,
            trajectories: List[List[ActionMaster]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:

        post_src, post_src_len = batch_dqn_input(
            trajectories, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id)

        actions, actions_lens, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)

        post_qs_target = self.target_sess.run(
            self.target_model.q_actions,
            feed_dict={
                self.target_model.src_: post_src,
                self.target_model.src_len_: post_src_len,
                self.target_model.actions_: actions,
                self.target_model.actions_len_: actions_lens,
                self.target_model.actions_repeats_: actions_repeats})

        post_qs_dqn = self.sess.run(
            self.model.q_actions,
            feed_dict={
                self.model.src_: post_src,
                self.model.src_len_: post_src_len,
                self.model.actions_: actions,
                self.model.actions_len_: actions_lens,
                self.model.actions_repeats_: actions_repeats})

        best_actions_idx = get_batch_best_1d_idx(post_qs_dqn, actions_repeats)
        best_qs = post_qs_target[best_actions_idx]
        expected_q = (
                np.asarray(rewards) +
                np.asarray(dones) * self.hp.final_gamma * best_qs)
        return expected_q

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

        pre_src, pre_src_len = batch_dqn_input(
            pre_trajectories, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id)
        (actions, actions_lens, actions_repeats, group_inv_valid_idx
         ) = batch_drrn_action_input(
            action_matrix, action_len, pre_action_mask)
        group_action_id = convert_real_id_to_group_id(
            action_idx, group_inv_valid_idx, actions_repeats)

        _, summaries, loss_eval, abs_loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={
                self.model.src_: pre_src,
                self.model.src_len_: pre_src_len,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: group_action_id,
                self.model.expected_q_: expected_q,
                self.model.actions_: actions,
                self.model.actions_len_: actions_lens,
                self.model.actions_repeats_: actions_repeats})

        return abs_loss
