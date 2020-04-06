from typing import Optional, List, Any, Tuple, Dict

import numpy as np

from deeptextworld.agents.base_agent import TFCore, ActionDesc, ACT_TYPE
from deeptextworld.agents.utils import ActionMaster, ObsInventory
from deeptextworld.agents.utils import batch_drrn_action_input
from deeptextworld.agents.utils import id_real2batch
from deeptextworld.agents.utils import get_best_batch_ids
from deeptextworld.agents.utils import get_best_1d_q
from deeptextworld.agents.utils import dqn_input
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
            action_mask: np.ndarray,
            cnt_action: Optional[Dict[int, float]]) -> ActionDesc:
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        """
        admissible_action_matrix = action_matrix[action_mask, :]
        admissible_action_len = action_len[action_mask]
        actions_repeats = [len(action_mask)]

        src, src_len, _ = self.trajectory2input(trajectory)
        q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
            self.model.src_: [src],
            self.model.src_len_: [src_len],
            self.model.actions_: admissible_action_matrix,
            self.model.actions_len_: admissible_action_len,
            self.model.actions_repeats_: actions_repeats
        })

        cnt_action_array = []
        for mid in action_mask:
            cnt_action_array.append(
                cnt_action[mid] if mid in cnt_action else 0.)

        action_idx, q_val = get_best_1d_q(q_actions_t - cnt_action_array)
        real_action_idx = action_mask[action_idx]
        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_drrn,
            action_idx=real_action_idx,
            token_idx=action_matrix[real_action_idx],
            action_len=action_len[real_action_idx],
            action=actions[real_action_idx],
            q_actions=q_actions_t)
        return action_desc

    def _compute_expected_q(
            self,
            action_mask: List[np.ndarray],
            trajectories: List[List[ActionMaster]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:

        post_src, post_src_len, _ = self.batch_trajectory2input(trajectories)
        actions, actions_lens, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)

        target_model, target_sess = self.get_target_model()
        post_qs_target = target_sess.run(
            target_model.q_actions,
            feed_dict={
                target_model.src_: post_src,
                target_model.src_len_: post_src_len,
                target_model.actions_: actions,
                target_model.actions_len_: actions_lens,
                target_model.actions_repeats_: actions_repeats})

        post_qs_dqn = self.sess.run(
            self.model.q_actions,
            feed_dict={
                self.model.src_: post_src,
                self.model.src_len_: post_src_len,
                self.model.actions_: actions,
                self.model.actions_len_: actions_lens,
                self.model.actions_repeats_: actions_repeats})

        best_actions_idx = get_best_batch_ids(post_qs_dqn, actions_repeats)
        best_qs = post_qs_target[best_actions_idx]
        expected_q = (
                np.asarray(rewards) +
                np.asarray(dones) * self.hp.gamma * best_qs)
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
            step: int,
            others: Any) -> np.ndarray:

        expected_q = self._compute_expected_q(
            post_action_mask, post_trajectories, action_matrix, action_len,
            dones, rewards)

        pre_src, pre_src_len, _ = self.batch_trajectory2input(pre_trajectories)
        (actions, actions_lens, actions_repeats, id_real2mask
         ) = batch_drrn_action_input(
            action_matrix, action_len, pre_action_mask)
        action_batch_ids = id_real2batch(
            action_idx, id_real2mask, actions_repeats)

        _, summaries, loss_eval, abs_loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={
                self.model.src_: pre_src,
                self.model.src_len_: pre_src_len,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: action_batch_ids,
                self.model.expected_q_: expected_q,
                self.model.actions_: actions,
                self.model.actions_len_: actions_lens,
                self.model.actions_repeats_: actions_repeats})

        self.train_summary_writer.add_summary(
            summaries, step - self.hp.observation_t)
        return abs_loss


class LegacyDRRNCore(DRRNCore):
    def trajectory2input(
            self, trajectory: List[ActionMaster]
    ) -> Tuple[List[int], int, List[int]]:
        return dqn_input(
            trajectory, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id, with_action_padding=True,
            max_action_size=self.hp.n_tokens_per_action)
