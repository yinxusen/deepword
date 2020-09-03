from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.agents.utils import *
from deeptextworld.utils import load_actions


class ZorkAgent(BaseAgent):
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
