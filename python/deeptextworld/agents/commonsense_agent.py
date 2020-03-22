import math
from typing import Optional, List, Any, Tuple, Dict

import numpy as np

from deeptextworld.agents.base_agent import ActionDesc, ACT_TYPE, TFCore
from deeptextworld.agents.utils import bert_commonsense_input, ActionMaster, \
    ObsInventory, dqn_input
from deeptextworld.models.export_models import CommonsenseModel


class BertCore(TFCore):
    """
    The agent that explores commonsense ability of BERT models.
    This agent combines each trajectory with all its actions together, separated
    with [SEP] in the middle. Then feeds the sentence into BERT to get a score
    from the [CLS] token.
    refer to https://arxiv.org/pdf/1810.04805.pdf for fine-tuning and evaluation
    """
    def __init__(self, hp, model_dir, tokenizer):
        super(BertCore, self).__init__(hp, model_dir, tokenizer)
        self.model: Optional[CommonsenseModel] = None
        self.target_model: Optional[CommonsenseModel] = None

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
        pass

    def trajectory2input(
            self, trajectory: List[ActionMaster]
    ) -> Tuple[List[int], int, List[int]]:
        # remove the length for [CLS] and two [SEP]s.
        return dqn_input(
            trajectory, self.tokenizer, self.hp.num_tokens - 3,
            self.hp.padding_val_id)

    def get_a_policy_action(
            self, trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray, action_len: np.ndarray,
            actions: List[str],
            action_mask: np.ndarray,
            cnt_action: Optional[Dict[int, float]]) -> ActionDesc:
        action_matrix = action_matrix[action_mask, :]
        action_len = action_len[action_mask]
        actions = np.asarray(actions)[action_mask]

        src, src_len, _ = self.trajectory2input(trajectory)
        inp, seg_tj_action, inp_size = bert_commonsense_input(
            action_matrix, action_len, src, src_len,
            self.hp.sep_val_id, self.hp.cls_val_id, self.hp.num_tokens)
        n_actions = inp.shape[0]
        self.debug("number of actions: {}".format(n_actions))
        # TODO: better allowed batch
        allowed_batch_size = 32
        n_batches = int(math.ceil(n_actions * 1. / allowed_batch_size))
        self.debug("compute q-values through {} batches".format(n_batches))
        total_q_actions = []
        for i in range(n_batches):
            ss = i * allowed_batch_size
            ee = min((i + 1) * allowed_batch_size, n_actions)
            q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
                self.model.src_: inp[ss: ee],
                self.model.seg_tj_action_: seg_tj_action[ss: ee],
                self.model.src_len_: inp_size[ss: ee],
            })
            total_q_actions.append(q_actions_t)

        q_actions_t = np.concatenate(total_q_actions, axis=-1)
        results = sorted(
            zip(list(actions), list(q_actions_t)), key=lambda x: x[-1])
        results = ["{}\t{}".format(a, q) for a, q in results]
        self.debug("\n".join(results))

        action_idx = np.argmax(q_actions_t)
        action = actions[action_idx]
        true_action_idx = action_mask[action_idx]

        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_drrn, action_idx=true_action_idx,
            token_idx=action_matrix[action_idx],
            action_len=action_len[action_idx],
            action=action)
        return action_desc
