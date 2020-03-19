from typing import List, Optional, Any

import numpy as np

from deeptextworld.agents.base_agent import ActionDesc, ACT_TYPE
from deeptextworld.agents.base_agent import BaseAgent, TFCore
from deeptextworld.agents.utils import get_best_2d_q, ActionMaster, ObsInventory
from deeptextworld.models.export_models import GenDQNModel


class GenDQNCore(TFCore):
    def __init__(self, hp, model_dir, tokenizer):
        super(GenDQNCore, self).__init__(hp, model_dir, tokenizer)
        self.model: Optional[GenDQNModel] = None
        self.target_model: Optional[GenDQNModel] = None

    def _compute_expected_q(
            self,
            trajectories: List[List[ActionMaster]],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:
        """
        Compute expected q values given post trajectories and post actions

        notice that action_mask, tids, sids should belong to post game states,
        while dones, rewards belong to pre game states.
        """

        src, src_len, master_mask = self.batch_trajectory2input(trajectories)
        target_model, target_sess = self.get_target_model()
        # target network provides the value used as expected q-values
        qs_target = target_sess.run(
            target_model.q_actions,
            feed_dict={
                target_model.src_: src,
                target_model.src_len_: src_len,
                target_model.src_seg_: master_mask})

        # current network decides which action provides best q-value
        qs_dqn = self.sess.run(
            self.model.q_actions,
            feed_dict={
                self.model.src_: src,
                self.model.src_len_: src_len,
                self.model.src_seg_: master_mask})

        expected_q = np.zeros_like(rewards)
        for i in range(len(expected_q)):
            expected_q[i] = rewards[i]
            if not dones[i]:
                s_argmax_q, _, valid_len = get_best_2d_q(
                    qs_dqn[i, :, :], self.hp.eos_id)
                expected_q[i] += self.hp.gamma * np.mean(
                    qs_target[i, range(valid_len), s_argmax_q[:valid_len]])

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
            b_weight: np.ndarray, step: int,
            others: Any) -> np.ndarray:

        expected_q = self._compute_expected_q(
            trajectories=post_trajectories, dones=dones, rewards=rewards)

        src, src_len, master_mask = self.batch_trajectory2input(
            pre_trajectories)

        action_token_ids = others
        at_id_wo_eos = np.asarray(action_token_ids)
        at_id_wo_eos[
            range(len(action_token_ids)), np.asarray(action_len) - 1] = 0
        at_id_in = np.concatenate(
            [np.asarray([[self.hp.sos_id]] * len(action_len)),
             at_id_wo_eos[:, :-1]], axis=1)

        action_token_ids = np.asarray(action_token_ids)
        self.debug("action in/out example:\n{} -- {}\n{} -- {}".format(
            at_id_in[0, :],
            self.tokenizer.convert_ids_to_tokens(
                at_id_in[0, :action_len[0]]),
            action_token_ids[0, :],
            self.tokenizer.convert_ids_to_tokens(
                action_token_ids[0, :action_len[0]])))

        _, summaries, loss_eval, abs_loss = self.sess.run(
            [self.model.train_op,
             self.model.train_summary_op,
             self.model.loss,
             self.model.abs_loss],
            feed_dict={
                self.model.src_: src,
                self.model.src_len_: src_len,
                self.model.src_seg_: master_mask,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: at_id_in,
                self.model.action_idx_out_: action_token_ids,
                self.model.action_len_: action_len,
                self.model.expected_q_: expected_q})

        return abs_loss

    def get_a_policy_action(
            self, trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray, action_len: np.ndarray,
            actions: List[str],
            action_mask: np.ndarray,
            cnt_action: Optional[np.ndarray]) -> ActionDesc:
        self.debug("trajectory: {}".format(trajectory))
        src, src_len, master_mask = self.trajectory2input(trajectory)
        self.debug("src: {}".format(src))
        self.debug("src_len: {}".format(src_len))
        self.debug("master_mask: {}".format(master_mask))
        beam_size = 1
        temperature = 1
        self.debug("temperature: {}".format(temperature))
        res = self.sess.run(
            [self.model.decoded_idx_infer,
             self.model.col_eos_idx,
             self.model.decoded_logits_infer,
             self.model.p_gen_infer],
            feed_dict={
                self.model.src_: [src],
                self.model.src_len_: [src_len],
                self.model.src_seg_: [master_mask],
                self.model.temperature_: temperature,
                self.model.beam_size_: beam_size,
                self.model.use_greedy_: False
            })
        action_idx = res[0]
        col_eos_idx = res[1]
        decoded_logits = res[2]
        p_gen = res[3]
        self.debug("decoded logits: {}".format(decoded_logits))

        res_summary = []
        special_tokens = {self.hp.padding_val, self.hp.eos}
        for bid in range(beam_size):
            action = " ".join(
                filter(lambda t: t not in special_tokens,
                       self.tokenizer.convert_ids_to_tokens(
                           action_idx[bid, :col_eos_idx[bid]])))
            res_summary.append(
                (action_idx[bid], col_eos_idx[bid],
                 action, p_gen[bid],
                 np.sum(decoded_logits[bid, :col_eos_idx[bid]])
                 / col_eos_idx[bid]))

        res_summary = list(reversed(sorted(res_summary, key=lambda x: x[-1])))
        top_action = res_summary[0]

        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_gen, action_idx=None,
            token_idx=top_action[0], action_len=top_action[1],
            action=top_action[2], q_actions=None)

        self.debug("generated actions:\n{}".format(
            "\n".join(
                [" ".join(
                    map(lambda a_p: "{}[{:.2f}]".format(a_p[0], a_p[1]),
                        zip(ac[2].split(), list(ac[3])))) + "\t{}".format(ac[4])
                 for ac in res_summary])))
        return action_desc


class GenDQNAgent(BaseAgent):
    def __init__(self, hp, model_dir):
        super(GenDQNAgent, self).__init__(hp, model_dir)

    def train_one_batch(self) -> None:
        """
        Train one batch of samples.
        Load target model if not exist, save current model when necessary.
        """
        # if there is not a well-trained model, it is unreasonable
        # to use target model.
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)

        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        action_id = [m.aid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        reward = [m.reward for m in b_memory]
        is_terminal = [m.is_terminal for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        next_action_mask = [m.next_action_mask for m in b_memory]
        action_token_ids = [m.token_id for m in b_memory]

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
            others=action_token_ids)

        self.memo.batch_update(b_idx, b_weight)

        if self.is_time_to_save():
            self.save_snapshot()
            self.core.save_model()
            self.core.create_or_reload_target_model()
