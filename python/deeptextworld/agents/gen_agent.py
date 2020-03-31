from collections import namedtuple
from typing import List, Optional, Any, Dict, Tuple

import numpy as np

from deeptextworld.agents.base_agent import ActionDesc, ACT_TYPE
from deeptextworld.agents.base_agent import BaseAgent, TFCore
from deeptextworld.agents.utils import ActionMaster, ObsInventory, Memolet
from deeptextworld.agents.utils import get_action_idx_pair
from deeptextworld.models.export_models import GenDQNModel


class GenSummary(namedtuple(
        "GenSummary", ("ids", "tokens", "gens", "q_action"))):

    def __repr__(self):
        return (" ".join(["{}[{:.2f}]".format(t, p)
                          for t, p in zip(self.tokens, self.gens)])
                + "\t{}".format(self.q_action))


class GenDQNCore(TFCore):
    def __init__(self, hp, model_dir, tokenizer):
        super(GenDQNCore, self).__init__(hp, model_dir, tokenizer)
        self.model: Optional[GenDQNModel] = None
        self.target_model: Optional[GenDQNModel] = None

    def concat_action2actions(
            self, concat_action_idx: List[int]) -> List[str]:
        concat_action = self.tokenizer.de_tokenize(concat_action_idx)
        actions = [a.strip() for a in concat_action.split(";")]
        return actions

    def gen_summary(
            self, action_idx: np.ndarray, col_eos_idx: np.ndarray,
            decoded_logits: np.ndarray, p_gen: np.ndarray, beam_size: int
    ) -> List[GenSummary]:
        """
        Return [ids, tokens, generation probabilities of each token, q_action]
        sorted by q_action (from larger to smaller)
        q_action: the average of decoded logits of selected tokens
        """
        res_summary = []
        for bid in range(beam_size):
            n_cols = col_eos_idx[bid]
            ids = list(action_idx[bid, :n_cols])
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            gen_prob_per_token = list(p_gen[bid, :n_cols])
            q_action = np.sum(decoded_logits[bid, :n_cols, ids]) / n_cols
            res_summary.append(
                GenSummary(ids, tokens, gen_prob_per_token, q_action))
        res_summary = list(reversed(sorted(res_summary, key=lambda x: x[-1])))
        return res_summary

    def get_a_policy_action(
            self, trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            actions: List[str],
            action_mask: np.ndarray,
            cnt_action: Optional[Dict[int, float]]) -> ActionDesc:
        self.debug("trajectory: {}".format(trajectory))
        src, src_len, master_mask = self.trajectory2input(trajectory)
        self.debug("src: {}".format(src))
        self.debug("src_len: {}".format(src_len))
        self.debug("master_mask: {}".format(master_mask))
        beam_size = 1
        temperature = 1
        use_greedy = False

        self.debug("use_greedy: {}, temperature: {}".format(
            use_greedy, temperature))
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
                self.model.use_greedy_: use_greedy
            })
        action_idx = res[0]
        col_eos_idx = res[1]
        decoded_logits = res[2]
        p_gen = res[3]

        res_summary = self.gen_summary(
            action_idx, col_eos_idx, decoded_logits, p_gen, beam_size)
        top_action = res_summary[0]
        self.debug("generated actions:\n{}".format(
            "\n".join([str(x) for x in res_summary])))

        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_gen, action_idx=None,
            token_idx=top_action[0], action_len=top_action[1],
            action=top_action[2], q_actions=None)

        self.debug("gen actions: {}".format(
            self.concat_action2actions(top_action.ids)))

        if self.hp.decode_concat_action:
            action_idx = np.random.choice(action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE.rnd,
                action_idx=action_idx,
                token_idx=action_matrix[action_idx],
                action_len=action_len[action_idx],
                action=actions[action_idx],
                q_actions=None)
        return action_desc

    def _compute_expected_q(
            self,
            trajectories: List[List[ActionMaster]],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:
        """
        Compute expected q values given post trajectories and post actions
        """

        src, src_len, master_mask = self.batch_trajectory2input(trajectories)
        target_model, target_sess = self.get_target_model()
        # target network provides the value used as expected q-values
        qs_target = target_sess.run(
            target_model.decoded_logits_infer,
            feed_dict={
                target_model.src_: src,
                target_model.src_len_: src_len,
                target_model.src_seg_: master_mask,
                target_model.beam_size_: 1,
                target_model.use_greedy_: True,
                target_model.temperature_: 1.
            })

        # current network decides which action provides best q-value
        s_argmax_q, valid_len = self.sess.run(
            [self.model.decoded_idx_infer, self.model.col_eos_idx],
            feed_dict={
                self.model.src_: src,
                self.model.src_len_: src_len,
                self.model.src_seg_: master_mask,
                self.model.beam_size_: 1,
                self.model.use_greedy_: True,
                self.model.temperature_: 1.})

        expected_q = np.zeros_like(rewards)
        for i in range(len(expected_q)):
            expected_q[i] = rewards[i]
            if not dones[i]:
                expected_q[i] += self.hp.gamma * np.mean(
                    qs_target[i, range(valid_len[i]),
                              s_argmax_q[i, :valid_len[i]]])

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
            trajectories=post_trajectories, dones=dones, rewards=rewards)

        src, src_len, master_mask = self.batch_trajectory2input(
            pre_trajectories)
        action_token_ids = others
        action_id_in, action_id_out, new_action_len = get_action_idx_pair(
            np.asarray(action_token_ids), np.asarray(action_len),
            self.hp.sos_id, self.hp.eos_id)
        self.debug("action in/out example:\n{} -- {}\n{} -- {}".format(
            action_id_in[0, :],
            self.tokenizer.de_tokenize(action_id_in[0, :]),
            action_id_out[0, :],
            self.tokenizer.de_tokenize(action_id_out[0, :])))

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
                self.model.action_idx_: action_id_in,
                self.model.action_idx_out_: action_id_out,
                self.model.action_len_: new_action_len,
                self.model.expected_q_: expected_q})
        self.train_summary_writer.add_summary(
            summaries, step - self.hp.observation_t)
        return abs_loss


class GenDQNAgent(BaseAgent):
    def __init__(self, hp, model_dir):
        super(GenDQNAgent, self).__init__(hp, model_dir)

    def _prepare_other_train_data(self, b_memory: List[Memolet]) -> Any:
        action_token_ids = [m.token_id for m in b_memory]
        return action_token_ids
