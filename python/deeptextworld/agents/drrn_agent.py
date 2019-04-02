import numpy as np
from textworld import EnvInfos

from deeptextworld import drrn_model
from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.dqn_func import get_random_1Daction, get_best_1Daction, get_best_1D_q
from deeptextworld.dqn_func import get_random_1Daction_fairly
from deeptextworld.utils import ctime


class DRRNAgent(BaseAgent):
    """
    """
    def __init__(self, hp, model_dir):
        super(DRRNAgent, self).__init__(hp, model_dir)

    def select_additional_infos(self) -> EnvInfos:
        """
        additional information needed when playing the game
        """
        request_infos = EnvInfos()
        if self.is_training:
            request_infos.description = True
            request_infos.inventory = True
            request_infos.entities = False
            request_infos.verbs = False
            request_infos.max_score = True
            request_infos.has_won = True
            request_infos.extras = ["recipe"]
            request_infos.admissible_commands = True
        else:
            request_infos.description = True
            request_infos.inventory = True
            request_infos.entities = False
            request_infos.verbs = False
            request_infos.max_score = True
            request_infos.has_won = True
            request_infos.has_lost = True
            request_infos.extras = ["recipe"]
            request_infos.admissible_commands = True
        return request_infos

    def get_an_eps_action(self, action_mask):
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param action_mask:
        """
        action_mask = self.fromBytes([action_mask])[0]
        reports = []
        if np.random.random() < self.eps:
            action_idx, player_t = get_random_1Daction(
                self.action_collector.get_actions(), action_mask)
            reports += [('random_action', action_idx),
                        ('action', player_t)]
        else:
            action_matrix = self.action_collector.get_action_matrix()
            actions_len = self.action_collector.get_action_len()
            indexed_state_t, lens_t = self.tjs.fetch_last_state()
            q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
                self.model.src_: [indexed_state_t],
                self.model.src_len_: [lens_t],
                self.model.actions_mask_: [action_mask],
                self.model.actions_: [action_matrix],
                self.model.actions_len_: [actions_len]
            })[0]
            action_idx, q_max, player_t = get_best_1Daction(
                q_actions_t, self.action_collector.get_actions(),
                mask=action_mask)
            reports += [('action', player_t), ('q_max', q_max),
                        ('q_argmax', action_idx)]
        return action_idx, player_t, reports

    def create_model_instance(self):
        model_creator = getattr(drrn_model, self.hp.model_creator)
        model = drrn_model.create_train_model(model_creator, self.hp)
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
        game_id = [m[0].gid for m in b_memory]
        reward = [m[0].reward for m in b_memory]
        is_terminal = [m[0].is_terminal for m in b_memory]
        action_mask = [m[0].action_mask for m in b_memory]

        next_idx, next_states = self.memo.get_next_states(b_idx)

        action_mask_t1 = (
            [self.zero_mask_bytes()
             if s.is_terminal else s.action_mask
             for s in next_states])

        action_mask_t = self.fromBytes(action_mask)
        action_mask_t1 = self.fromBytes(action_mask_t1)

        p_states, s_states, p_len, s_len =\
            self.tjs.fetch_batch_states_pair(trajectory_id, state_id)

        action_matrix = [self.action_collector.get_action_matrix(i) for i in game_id]
        actions_len = [self.action_collector.get_action_len(i) for i in game_id]

        t2 = ctime()
        s_q_actions_target = target_sess.run(
            self.target_model.q_actions,
            feed_dict={self.target_model.src_: s_states,
                       self.target_model.src_len_: s_len,
                       self.target_model.actions_: action_matrix,
                       self.target_model.actions_len_: actions_len,
                       self.target_model.actions_mask_: action_mask_t1})

        s_q_actions_dqn = sess.run(
            self.model.q_actions,
            feed_dict={self.model.src_: s_states,
                       self.model.src_len_: s_len,
                       self.model.actions_: action_matrix,
                       self.model.actions_len_: actions_len,
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
                       self.model.src_len_: p_len,
                       self.model.b_weight_: b_weight,
                       self.model.action_idx_: action_id,
                       self.model.actions_mask_: action_mask_t,
                       self.model.expected_q_: expected_q,
                       self.model.actions_: action_matrix,
                       self.model.actions_len_: actions_len})
        t3_end = ctime()

        self.memo.batch_update(b_idx, abs_loss)

        self.info('loss: {}'.format(loss_eval))
        self.debug('t1: {}, t2: {}, t3: {}'.format(t1_end-t1, t2_end-t2, t3_end-t3))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)
