from typing import Optional, List, Any, Tuple

import numpy as np

from deeptextworld.agents.base_agent import TFCore, ActionDesc, ACT_TYPE
from deeptextworld.agents.utils import batch_drrn_action_input
from deeptextworld.agents.utils import get_batch_best_1d_idx
from deeptextworld.agents.utils import convert_real_id_to_group_id
from deeptextworld.agents.utils import action_master2str
from deeptextworld.agents.utils import ActionMaster, ObsInventory
from deeptextworld.agents.utils import get_best_1d_q
from deeptextworld.models.export_models import DRRNModel
from deeptextworld.utils import flatten


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
            cnt_action: Optional[np.ndarray]) -> ActionDesc:
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        """
        mask_idx = np.where(action_mask == 1)[0]
        admissible_action_matrix = action_matrix[mask_idx, :]
        admissible_action_len = action_len[mask_idx]
        actions_repeats = [len(mask_idx)]

        src, src_len = self.trajectory2input(trajectory)
        self.debug("trajectory: {}".format(trajectory))
        self.debug("src: {}".format(src))
        self.debug("src_len: {}".format(src_len))
        self.debug("action_matrix: {}".format(action_matrix))
        self.debug("admissible_action_matrix: {}".format(admissible_action_matrix))
        q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
            self.model.src_: [src],
            self.model.src_len_: [src_len],
            self.model.actions_: admissible_action_matrix,
            self.model.actions_len_: admissible_action_len,
            self.model.actions_repeats_: actions_repeats
        })
        self.debug("q_actions_t {}".format(q_actions_t))
        action_idx, q_val = get_best_1d_q(q_actions_t - cnt_action[mask_idx])
        # action_idx, q_val = get_best_1d_q(q_actions_t)
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

        post_src, post_src_len = self.batch_trajectory2input(trajectories)
        actions, actions_lens, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)

        if self.target_model is None:
            target_model = self.model
            target_sess = self.sess
        else:
            target_model = self.target_model
            target_sess = self.target_sess

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

        pre_src, pre_src_len = self.batch_trajectory2input(pre_trajectories)
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


class LegacyDRRNCore(DRRNCore):
    def pad_action(self, action_tokens: List[str]) -> List[str]:
        if 0 < len(action_tokens) < self.hp.n_tokens_per_action:
            return (action_tokens + [self.hp.padding_val]
                    * (self.hp.n_tokens_per_action - len(action_tokens)))
        else:
            return action_tokens

    def trajectory2input(
            self, trajectory: List[ActionMaster]) -> Tuple[List[int], int]:
        tokens = []
        for am in trajectory:
            tokens += self.pad_action(
                self.tokenizer.tokenize(am.action))
            tokens += self.tokenizer.tokenize(am.master)

        trajectory_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        padding_size = self.hp.num_tokens - len(trajectory_ids)
        if padding_size >= 0:
            src = trajectory_ids + [self.hp.padding_val_id] * padding_size
            src_len = len(trajectory_ids)
        else:
            src = trajectory_ids[-padding_size:]
            src_len = self.hp.num_tokens
        return src, src_len
