"""
We generate admissible actions at every step, and then use DRRN to choose
the best action to play.

This agent can be compared with previous template-gen agent.
"""

import os
from os.path import join as pjoin

from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.agents.competition_agent import CompetitionAgent
from deeptextworld.agents.gen_agent import GenDQNCore
from deeptextworld.agents.utils import ACT
from deeptextworld.hparams import load_hparams

home_dir = os.path.expanduser("~")
gen_model_dir = pjoin(
    home_dir,
    "experiments-drrn/agent-student-train-gen-student-model-pre-train")


class GenDRRNAgent(BaseAgent):
    def __init__(self, hp, model_dir):
        super(GenDRRNAgent, self).__init__(hp, model_dir)
        gen_hp = load_hparams(
            fn_model_config="{}/hparams.json".format(gen_model_dir))
        gen_hp, gen_tokenizer = self.init_tokens(gen_hp)
        self.gen_core = GenDQNCore(gen_hp, gen_model_dir, gen_tokenizer)

    def _init_impl(self, load_best=False, restore_from=None) -> None:
        super(GenDRRNAgent, self)._init_impl(load_best, restore_from)
        self.gen_core.init(
            is_training=False, load_best=False, restore_from=None)

    def get_admissible_actions(self, infos):
        trajectory = self.tjs.fetch_last_state()
        actions = self.gen_core.get_decoded_concat_actions(trajectory)
        actions = list(set(actions + [ACT.look, ACT.inventory]))
        return actions


class GenCompetitionDRRNAgent(CompetitionAgent):
    def __init__(self, hp, model_dir):
        super(GenCompetitionDRRNAgent, self).__init__(hp, model_dir)
        gen_hp = load_hparams(
            fn_model_config="{}/hparams.json".format(gen_model_dir))
        gen_hp, gen_tokenizer = self.init_tokens(gen_hp)
        self.gen_core = GenDQNCore(gen_hp, gen_model_dir, gen_tokenizer)

    def _init_impl(self, load_best=False, restore_from=None) -> None:
        super(GenCompetitionDRRNAgent, self)._init_impl(load_best, restore_from)
        self.gen_core.init(
            is_training=False, load_best=False, restore_from=None)

    def get_admissible_actions(self, infos):
        trajectory = self.tjs.fetch_last_state()
        actions = self.gen_core.get_decoded_concat_actions(trajectory)
        actions = list(set(actions + [ACT.look, ACT.inventory]))
        return actions
