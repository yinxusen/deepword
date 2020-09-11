from os import path

from deepword.agents.base_agent import BaseAgent
from deepword.agents.competition_agent import CompetitionAgent
from deepword.agents.cores import PGNCore
from deepword.agents.utils import ACT
from deepword.hparams import load_hparams
from deepword.tokenizers import init_tokens

home_dir = path.expanduser("~")
gen_model_dir = path.join(
    home_dir,
    "experiments-drrn/agent-student-train-gen-student-model-pre-train")


class GenDRRNAgent(BaseAgent):
    """
    We generate admissible actions at every step, and then use DRRN to choose
    the best action to play.

    This agent can be compared with previous template-gen agent.
    """

    def __init__(self, hp, model_dir):
        super(GenDRRNAgent, self).__init__(hp, model_dir)
        gen_hp = load_hparams(
            fn_model_config="{}/hparams.json".format(gen_model_dir))
        gen_hp, gen_tokenizer = init_tokens(gen_hp)
        self.gen_core = PGNCore(gen_hp, gen_model_dir, gen_tokenizer)

    def _init_impl(self, load_best=False, restore_from=None) -> None:
        super(GenDRRNAgent, self)._init_impl(load_best, restore_from)
        # colocate gen-core and core on the same GPU
        self.gen_core.set_d4eval(self.core.d4eval)
        self.gen_core.init(
            is_training=False, load_best=False, restore_from=None)

    def _get_admissible_actions(self, infos):
        trajectory = self.tjs.fetch_last_state()
        actions = self.gen_core.generate_admissible_actions(trajectory)
        actions = list(set(actions + [ACT.look, ACT.inventory]))
        return actions


class GenCompetitionDRRNAgent(CompetitionAgent):
    def __init__(self, hp, model_dir):
        super(GenCompetitionDRRNAgent, self).__init__(hp, model_dir)
        gen_hp = load_hparams(
            fn_model_config="{}/hparams.json".format(gen_model_dir))
        gen_hp, gen_tokenizer = init_tokens(gen_hp)
        self.gen_core = PGNCore(gen_hp, gen_model_dir, gen_tokenizer)

    def _init_impl(self, load_best=False, restore_from=None) -> None:
        super(GenCompetitionDRRNAgent, self)._init_impl(load_best, restore_from)
        # colocate gen-core and core on the same GPU
        self.gen_core.set_d4eval(self.core.d4eval)
        self.gen_core.init(
            is_training=False, load_best=False, restore_from=None)

    def _get_admissible_actions(self, infos):
        trajectory = self.tjs.fetch_last_state()
        actions = self.gen_core.generate_admissible_actions(trajectory)
        actions = list(set(actions + [ACT.look, ACT.inventory]))
        return actions
