import hashlib
import numpy as np
from textworld import EnvInfos

from deeptextworld import dqn_model
from deeptextworld.agents.base_agent import BaseAgent, ActionDesc, \
    ACT_TYPE_RND_CHOOSE, ACT_TYPE_NN, ACT_TYPE_TBL
from deeptextworld.dqn_func import get_random_1Daction, get_best_1Daction, \
    get_best_1D_q
from deeptextworld.utils import ctime


class DQNAgent(BaseAgent):
    """
    """
    def __init__(self, hp, model_dir):
        super(DQNAgent, self).__init__(hp, model_dir)

    def select_additional_infos(self):
        """
        additional information needed when playing the game
        """
        return EnvInfos(
            description=True,
            inventory=True,
            max_score=True,
            has_won=True,
            admissible_commands=True,
            extras=['recipe'])

    def get_an_eps_action(self, action_mask):
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param action_mask:
        """
        action_mask = self.from_bytes([action_mask])[0]
        if np.random.random() < self.eps:
            action_idx, action = get_random_1Daction(
                self.action_collector.get_actions(), action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE_RND_CHOOSE, action_idx=action_idx,
                action=action)
        else:
            indexed_state_t, lens_t = self.tjs.fetch_last_state()
            q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
                self.model.src_: [indexed_state_t],
                self.model.src_len_: [lens_t]
            })[0]
            action_idx, q_max, action = get_best_1Daction(
                q_actions_t, self.action_collector.get_actions(),
                mask=action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE_NN, action_idx=action_idx, action=action)
        return action_desc

    def create_model_instance(self):
        model_creator = getattr(dqn_model, self.hp.model_creator)
        model = dqn_model.create_train_model(model_creator, self.hp)
        return model

    def create_eval_model_instance(self):
        model_creator = getattr(dqn_model, self.hp.model_creator)
        model = dqn_model.create_eval_model(model_creator, self.hp)
        return model

    def train_impl(self, sess, t, summary_writer, target_sess):
        gamma = self.reverse_annealing_gamma(
            self.hp.init_gamma, self.hp.final_gamma,
            t - self.hp.observation_t, self.hp.annealing_gamma_t)

        t1 = ctime()
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)
        t1_end = ctime()

        trajectory_id = [m[0].tid for m in b_memory]
        state_id = [m[0].sid for m in b_memory]
        action_id = [m[0].aid for m in b_memory]
        reward = [m[0].reward for m in b_memory]
        is_terminal = [m[0].is_terminal for m in b_memory]
        next_action_mask = [m[0].next_action_mask for m in b_memory]

        action_mask_t1 = self.from_bytes(next_action_mask)

        p_states, s_states, p_len, s_len =\
            self.tjs.fetch_batch_states_pair(trajectory_id, state_id)

        t2 = ctime()
        s_q_actions_target = target_sess.run(
            self.target_model.q_actions,
            feed_dict={self.target_model.src_: s_states,
                       self.target_model.src_len_: s_len})

        s_q_actions_dqn = sess.run(
            self.model.q_actions,
            feed_dict={self.model.src_: s_states,
                       self.model.src_len_: s_len})
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
                       self.model.src_len_: p_len,
                       self.model.b_weight_: b_weight,
                       self.model.action_idx_: action_id,
                       self.model.expected_q_: expected_q})
        t3_end = ctime()

        self.memo.batch_update(b_idx, abs_loss)

        self.info('loss: {}'.format(loss_eval))
        self.debug('t1: {}, t2: {}, t3: {}'.format(
            t1_end-t1, t2_end-t2, t3_end-t3))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)


class TabularDQNAgent(DQNAgent):
    def __init__(self, hp, model_dir):
        super(TabularDQNAgent, self).__init__(hp, model_dir)

    def get_an_eps_action(self, action_mask):
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param action_mask:
        """
        action_mask = self.from_bytes([action_mask])[0]
        if np.random.random() < self.eps:
            action_idx, action = get_random_1Daction(
                self.action_collector.get_actions(), action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE_RND_CHOOSE, action_idx=action_idx,
                action=action)
        else:
            state_text, len_state_text = self.stc.fetch_last_state()
            q_actions_t = self.q_mat.get(
                self.get_hash(state_text), np.zeros(self.hp.n_actions))
            action_idx, q_max, action = get_best_1Daction(
                q_actions_t, self.action_collector.get_actions(),
                mask=action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE_TBL, action_idx=action_idx, action=action)
        return action_desc

    @classmethod
    def get_hash(cls, txt):
        return hashlib.md5(txt.encode("utf-8")).hexdigest()

    def train_impl(self, sess, t, summary_writer, target_sess):
        gamma = self.reverse_annealing_gamma(
            self.hp.init_gamma, self.hp.final_gamma,
            t - self.hp.observation_t, self.hp.annealing_gamma_t)

        t1 = ctime()
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)
        t1_end = ctime()

        trajectory_id = [m[0].tid for m in b_memory]
        state_id = [m[0].sid for m in b_memory]
        action_id = [m[0].aid for m in b_memory]
        reward = [m[0].reward for m in b_memory]
        is_terminal = [m[0].is_terminal for m in b_memory]
        next_action_mask = [m[0].next_action_mask for m in b_memory]

        action_mask_t1 = self.from_bytes(next_action_mask)

        p_states, s_states, p_len, s_len =\
            self.stc.fetch_batch_states_pair(trajectory_id, state_id)

        t2 = ctime()
        s_q_actions_target = np.asarray(
            [self.q_mat.get(self.get_hash(s), np.zeros(self.hp.n_actions))
             for s in s_states])
        s_q_actions_dqn = np.asarray(
            [self.q_mat.get(self.get_hash(s), np.zeros(self.hp.n_actions))
             for s in s_states])
        t2_end = ctime()

        expected_q = np.zeros_like(reward)
        for i in range(len(expected_q)):
            expected_q[i] = reward[i]
            if not is_terminal[i]:
                s_argmax_q, _ = get_best_1D_q(
                    s_q_actions_dqn[i, :], mask=action_mask_t1[i])
                expected_q[i] += gamma * s_q_actions_target[i, s_argmax_q]

        t3 = ctime()
        for i, s in enumerate(s_states):
            if self.get_hash(s) not in self.q_mat:
                self.q_mat[self.get_hash(s)] = np.zeros(self.hp.n_actions)
            self.q_mat[self.get_hash(s)][action_id[i]] = expected_q[i]
        t3_end = ctime()

        self.debug('t1: {}, t2: {}, t3: {}'.format(
            t1_end-t1, t2_end-t2, t3_end-t3))
