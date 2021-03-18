from deepword.agents.base_agent import BaseAgent
from deepword.agents.utils import *
from deepword.hparams import conventions
from deepword.utils import load_actions


class ZorkAgent(BaseAgent):
    """
    The agent to run Zork.

    TextWorld will not provide admissible actions like cooking games, so a
     loaded action file is required.
    """
    def __init__(self, hp, model_dir):
        super(ZorkAgent, self).__init__(hp, model_dir)
        assert self.hp.action_file is not None, "action file is needed"
        self.loaded_actions: List[str] = [
            a.lower() for a in load_actions(self.hp.action_file)]

    def _get_admissible_actions(self, infos):
        """
        We add inventory and look, in case that the game doesn't provide these
        two key actions.
        TODO: original code for playing Zork doesn't have "look"
        TODO: for DQN agent, the order of actions matters, make sure not using
            set or other operators that changing orders.
        """
        admissible_actions = self.loaded_actions
        return admissible_actions

    @classmethod
    def read_expert_guides(cls, fn_expert_guides):
        with open(fn_expert_guides, "r") as f:
            lines = [x.strip() for x in f.readlines()]
        return lines

    def _start_episode_impl(self, obs, infos):
        super(ZorkAgent, self)._start_episode_impl(obs, infos)
        # load Zork walkthrough
        if self.hp.walkthrough_guided_exploration:
            self._walkthrough = self.read_expert_guides(
                conventions.zork1_walkthrough)
            self._continue_walkthrough = True
