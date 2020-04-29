"""
We generate admissible actions at every step, and then use DRRN to choose
the best action to play.

This agent can be compared with previous template-gen agent.
"""

import os
from os.path import join as pjoin
from typing import List, Optional, Any, Dict

import numpy as np

from deeptextworld.agents.base_agent import ActionDesc
from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.agents.base_agent import TFCore
from deeptextworld.agents.competition_agent import CompetitionAgent
from deeptextworld.agents.utils import ACT, GenSummary
from deeptextworld.agents.utils import ActionMaster, ObsInventory
from deeptextworld.hparams import load_hparams
from deeptextworld.models.export_models import GenDQNModel

home_dir = os.path.expanduser("~")
gen_model_dir = pjoin(
    home_dir,
    "experiments-drrn/agent-student-train-gen-student-model-pre-train")


class PGNCore(TFCore):
    """Generate admissible actions for games, given only trajectory"""
    def __init__(self, hp, model_dir, tokenizer):
        super(PGNCore, self).__init__(hp, model_dir, tokenizer)
        self.model: Optional[GenDQNModel] = None

    def summary(
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
                GenSummary(ids, tokens, gen_prob_per_token, q_action, n_cols))
        res_summary = list(reversed(sorted(res_summary, key=lambda x: x[-1])))
        return res_summary

    def decode(
            self, trajectory: List[ActionMaster], beam_size: int,
            temperature: float, use_greedy: bool) -> List[GenSummary]:
        src, src_len, master_mask = self.trajectory2input(trajectory)
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
        res_summary = self.summary(
            action_idx, col_eos_idx, decoded_logits, p_gen, beam_size)
        self.debug("generated results:\n{}".format(
            "\n".join([str(x) for x in res_summary])))
        return res_summary

    def generate_admissible_actions(
            self, trajectory: List[ActionMaster]) -> List[str]:

        if self.hp.decode_concat_action:
            res = self.decode(
                trajectory, beam_size=1, temperature=1., use_greedy=False)
            concat_actions = self.tokenizer.de_tokenize(res[0].ids)
            actions = [a.strip() for a in concat_actions.split(";")]
        else:
            res = self.decode(
                trajectory, beam_size=20, temperature=1., use_greedy=False)
            actions = [self.tokenizer.de_tokenize(x.ids) for x in res]
        return actions

    def get_a_policy_action(
            self, trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            actions: List[str],
            action_mask: np.ndarray,
            cnt_action: Optional[Dict[int, float]]) -> ActionDesc:
        raise NotImplementedError()

    def train_one_batch(
            self, pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray], dones: List[bool],
            rewards: List[float], action_idx: List[int],
            b_weight: np.ndarray, step: int,
            others: Any) -> np.ndarray:
        raise NotImplementedError()


class GenDRRNAgent(BaseAgent):
    def __init__(self, hp, model_dir):
        super(GenDRRNAgent, self).__init__(hp, model_dir)
        gen_hp = load_hparams(
            fn_model_config="{}/hparams.json".format(gen_model_dir))
        gen_hp, gen_tokenizer = self.init_tokens(gen_hp)
        self.gen_core = PGNCore(gen_hp, gen_model_dir, gen_tokenizer)

    def _init_impl(self, load_best=False, restore_from=None) -> None:
        super(GenDRRNAgent, self)._init_impl(load_best, restore_from)
        # colocate gen-core and core on the same GPU
        self.gen_core.set_d4eval(self.core.d4eval)
        self.gen_core.init(
            is_training=False, load_best=False, restore_from=None)

    def get_admissible_actions(self, infos):
        trajectory = self.tjs.fetch_last_state()
        actions = self.gen_core.generate_admissible_actions(trajectory)
        actions = list(set(actions + [ACT.look, ACT.inventory]))
        return actions


class GenCompetitionDRRNAgent(CompetitionAgent):
    def __init__(self, hp, model_dir):
        super(GenCompetitionDRRNAgent, self).__init__(hp, model_dir)
        gen_hp = load_hparams(
            fn_model_config="{}/hparams.json".format(gen_model_dir))
        gen_hp, gen_tokenizer = self.init_tokens(gen_hp)
        self.gen_core = PGNCore(gen_hp, gen_model_dir, gen_tokenizer)

    def _init_impl(self, load_best=False, restore_from=None) -> None:
        super(GenCompetitionDRRNAgent, self)._init_impl(load_best, restore_from)
        # colocate gen-core and core on the same GPU
        self.gen_core.set_d4eval(self.core.d4eval)
        self.gen_core.init(
            is_training=False, load_best=False, restore_from=None)

    def get_admissible_actions(self, infos):
        trajectory = self.tjs.fetch_last_state()
        actions = self.gen_core.generate_admissible_actions(trajectory)
        actions = list(set(actions + [ACT.look, ACT.inventory]))
        return actions
