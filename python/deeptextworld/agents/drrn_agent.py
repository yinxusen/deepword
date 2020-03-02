import math

import numpy as np

from deeptextworld.agents.base_agent import BaseAgent, ActionDesc, ACT_TYPE
from deeptextworld.models import drrn_model
from deeptextworld.models.dqn_func import get_random_1Daction, \
    get_best_1Daction, get_best_1D_q
from deeptextworld.agents.utils import bert_commonsense_input
from deeptextworld.utils import ctime


class DRRNAgent(BaseAgent):
    """
    """
    def __init__(self, hp, model_dir):
        super(DRRNAgent, self).__init__(hp, model_dir)

    def get_an_eps_action(self, action_mask):
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param action_mask:
        """
        action_mask = self.from_bytes([action_mask])[0]
        if np.random.random() < self.eps:
            action_idx, action = get_random_1Daction(
                self.actor.actions, action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE.rnd, action_idx=action_idx,
                token_idx=self.actor.action_matrix[action_idx],
                action_len=self.actor.action_len[action_idx],
                action=action)
        else:
            action_matrix = self.actor.action_matrix
            action_len = self.actor.action_len
            indexed_state_t, lens_t = self.tjs.fetch_last_state()
            seg_t, _ = self.tjs_seg.fetch_last_state()
            q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
                self.model.src_: [indexed_state_t],
                self.model.src_seg_: [seg_t],
                self.model.src_len_: [lens_t],
                self.model.actions_mask_: [action_mask],
                self.model.actions_: [action_matrix],
                self.model.actions_len_: [action_len]
            })[0]
            actions = self.actor.actions
            action_idx, q_max, action = get_best_1Daction(
                q_actions_t - self._cnt_action, actions,
                mask=action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE.policy_drrn, action_idx=action_idx,
                token_idx=self.actor.action_matrix[action_idx],
                action_len=self.actor.action_len[action_idx],
                action=action)
        return action_desc

    def create_model_instance(self, device):
        model_creator = getattr(drrn_model, self.hp.model_creator)
        model = drrn_model.create_train_model(model_creator, self.hp)
        return model

    def create_eval_model_instance(self, device):
        model_creator = getattr(drrn_model, self.hp.model_creator)
        model = drrn_model.create_eval_model(model_creator, self.hp)
        return model

    def train_impl(self, sess, t, summary_writer, target_sess, target_model):
        gamma = self.reverse_annealing_gamma(
            self.hp.init_gamma, self.hp.final_gamma,
            t - self.hp.observation_t, self.hp.annealing_gamma_t)

        t1 = ctime()
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)
        t1_end = ctime()

        trajectory_id = [m[0].tid for m in b_memory]
        state_id = [m[0].sid for m in b_memory]
        action_id = [m[0].aid for m in b_memory]
        game_id = [m[0].gid for m in b_memory]
        reward = [m[0].reward for m in b_memory]
        is_terminal = [m[0].is_terminal for m in b_memory]
        action_mask = [m[0].action_mask for m in b_memory]
        next_action_mask = [m[0].next_action_mask for m in b_memory]

        action_mask_t = self.from_bytes(action_mask)
        action_mask_t1 = self.from_bytes(next_action_mask)

        p_states, s_states, p_len, s_len =\
            self.tjs.fetch_batch_states_pair(trajectory_id, state_id)

        p_seg, s_seg, _, _ = \
            self.tjs_seg.fetch_batch_states_pair(trajectory_id, state_id)

        # make sure the p_states and s_states are in the same game.
        # otherwise, it won't make sense to use the same action matrix.
        action_len = (
            [self.actor.get_action_len(gid) for gid in game_id])
        max_action_len = np.max(action_len)
        action_matrix = (
            [self.actor.get_action_matrix(gid)[:, :max_action_len]
             for gid in game_id])

        t2 = ctime()
        s_q_actions_target = target_sess.run(
            target_model.q_actions,
            feed_dict={target_model.src_: s_states,
                       target_model.src_seg_: s_seg,
                       target_model.src_len_: s_len,
                       target_model.actions_: action_matrix,
                       target_model.actions_len_: action_len,
                       target_model.actions_mask_: action_mask_t1})

        s_q_actions_dqn = sess.run(
            self.model.q_actions,
            feed_dict={self.model.src_: s_states,
                       self.model.src_seg_: s_seg,
                       self.model.src_len_: s_len,
                       self.model.actions_: action_matrix,
                       self.model.actions_len_: action_len,
                       self.model.actions_mask_: action_mask_t1})
        t2_end = ctime()

        expected_q = np.zeros_like(reward)
        for i in range(len(expected_q)):
            expected_q[i] = reward[i]
            if not is_terminal[i]:
                s_argmax_q, _ = get_best_1D_q(
                    s_q_actions_dqn[i, :], mask=action_mask_t1[i])
                expected_q[i] += gamma * s_q_actions_target[i, s_argmax_q]

        t3 = ctime()
        _, summaries, loss_eval, abs_loss = sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={self.model.src_: p_states,
                       self.model.src_seg_: p_seg,
                       self.model.src_len_: p_len,
                       self.model.b_weight_: b_weight,
                       self.model.action_idx_: action_id,
                       self.model.actions_mask_: action_mask_t,
                       self.model.expected_q_: expected_q,
                       self.model.actions_: action_matrix,
                       self.model.actions_len_: action_len})
        t3_end = ctime()

        self.memo.batch_update(b_idx, abs_loss)
        if t % 1000 == 0:
            self.debug('t: {}, t1: {}, t2: {}, t3: {}'.format(
                t, t1_end-t1, t2_end-t2, t3_end-t3))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)


class BertDRRNAgent(DRRNAgent):
    def __init__(self, hp, model_dir):
        super(BertDRRNAgent, self).__init__(hp, model_dir)


class BertAgent(BaseAgent):
    """
    The agent that explores commonsense ability of BERT models.
    This agent combines each trajectory with all its actions together, separated
    with [SEP] in the middle. Then feeds the sentence into BERT to get a score
    from the [CLS] token.
    refer to https://arxiv.org/pdf/1810.04805.pdf for fine-tuning and evaluation
    """
    def __init__(self, hp, model_dir):
        super(BertAgent, self).__init__(hp, model_dir)

    def _init_impl(self, load_best=False, restore_from=None):
        super(BertAgent, self)._init_impl(load_best, restore_from)
        # for Bert commonsense model, we combine
        # [trajectory], [SEP], [action], [SEP]
        # as a input sentence, so we need to subtract two [SEP]s and [action]
        # from maximum allowed number of tokens.
        self.tjs.num_tokens = (
            self.hp.num_tokens - self.hp.n_tokens_per_action - 2)

    def get_an_eps_action(self, action_mask):
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param action_mask:
        """
        action_mask = self.from_bytes([action_mask])[0]
        mask_idx = np.where(action_mask == 1)[0]
        action_matrix = self.actor.action_matrix[mask_idx, :]
        action_len = self.actor.action_len[mask_idx]
        actions = np.asarray(self.actor.actions)[mask_idx]

        if np.random.random() < self.eps:
            action_idx, action = get_random_1Daction(actions)
            true_action_idx = mask_idx[action_idx]
            action_desc = ActionDesc(
                action_type=ACT_TYPE.rnd, action_idx=true_action_idx,
                token_idx=action_matrix[action_idx],
                action_len=action_len[action_idx],
                action=action)
        else:
            indexed_state_t, lens_t = self.tjs.fetch_last_state()
            inp, seg_tj_action, inp_size = bert_commonsense_input(
                action_matrix, action_len, indexed_state_t, lens_t,
                self.hp.sep_val_id, self.hp.num_tokens)
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
            true_action_idx = mask_idx[action_idx]

            action_desc = ActionDesc(
                action_type=ACT_TYPE.policy_drrn, action_idx=true_action_idx,
                token_idx=action_matrix[action_idx],
                action_len=action_len[action_idx],
                action=action)
        return action_desc

    def create_model_instance(self, device):
        model_creator = getattr(drrn_model, self.hp.model_creator)
        model = model_creator.get_train_model(self.hp, device)
        return model

    def create_eval_model_instance(self, device):
        model_creator = getattr(drrn_model, self.hp.model_creator)
        model = model_creator.get_eval_model(self.hp, device)
        return model

    def train_impl(self, sess, t, summary_writer, target_sess, target_model):
        raise NotImplementedError()
