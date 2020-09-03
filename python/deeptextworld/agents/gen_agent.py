from typing import List, Any

import numpy as np

from deeptextworld.agents.base_agent import ActionDesc, ACT_TYPE
from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.agents.utils import Memolet


class GenDQNAgent(BaseAgent):
    """
    GenDQNAgent works with :py:class:`deeptextworld.agents.cores.GenDQNCore`.
    """
    def __init__(self, hp, model_dir):
        super(GenDQNAgent, self).__init__(hp, model_dir)

    def _prepare_other_train_data(self, b_memory: List[Memolet]) -> Any:
        action_token_ids = [m.token_id for m in b_memory]
        return action_token_ids

    def _get_policy_action(self, action_mask: np.ndarray) -> ActionDesc:
        trajectory = self.tjs.fetch_last_state()
        gen_res = self.core.decode_action(trajectory)
        action = self.tokenizer.de_tokenize(gen_res.ids)
        self.debug("gen action: {}".format(action))
        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_gen,
            action_idx=None,
            token_idx=gen_res.ids,
            action_len=gen_res.len,
            action=action,
            q_actions=gen_res.q_action)
        return action_desc
