import numpy as np
from bert.tokenization import FullTokenizer

from deeptextworld import dsqn_model
from deeptextworld.agents.base_agent import BaseAgent, ActionDesc, \
    ACT_TYPE_RND_CHOOSE, ACT_TYPE_NN
from deeptextworld.dqn_func import get_random_1Daction, get_best_1Daction, \
    get_best_1D_q
from deeptextworld.hparams import copy_hparams
from deeptextworld.utils import ctime, load_vocab, get_token2idx


class DSQNAgent(BaseAgent):
    """
    """
    def __init__(self, hp, model_dir):
        super(DSQNAgent, self).__init__(hp, model_dir)

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
            action_matrix = self.action_collector.get_action_matrix()
            action_len = self.action_collector.get_action_len()
            indexed_state_t, lens_t = self.tjs.fetch_last_state()
            q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
                self.model.src_: [indexed_state_t],
                self.model.src_len_: [lens_t],
                self.model.actions_mask_: [action_mask],
                self.model.actions_: [action_matrix],
                self.model.actions_len_: [action_len]
            })[0]
            actions = self.action_collector.get_actions()
            action_idx, q_max, action = get_best_1Daction(
                q_actions_t - self._cnt_action, actions,
                mask=action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE_NN, action_idx=action_idx, action=action)
        return action_desc

    def create_model_instance(self):
        model_creator = getattr(dsqn_model, self.hp.model_creator)
        model = dsqn_model.create_train_model(model_creator, self.hp)
        return model

    def create_eval_model_instance(self):
        model_creator = getattr(dsqn_model, self.hp.model_creator)
        model = dsqn_model.create_eval_model(model_creator, self.hp)
        return model

    def get_train_pairs(self, tids, sids):
        sids = np.asarray(sids)
        # we don't need the terminal state
        # also the terminal state is not used in the hash_states2tjs
        tj, tj_len = \
            self.tjs.fetch_batch_states(tids, sids - 1)
        st, st_len = \
            self.stc.fetch_batch_states(tids, sids - 1)
        hs = list(map(lambda txt: self.get_hash(txt), st))
        same_states = []
        diff_states = []
        all_states = list(self.hash_states2tjs.keys())
        for txt in st:
            hs = self.get_hash(txt)
            sampled_idx = np.random.choice(
                list(range(len(self.hash_states2tjs[hs]))))
            same_states.append(self.hash_states2tjs[hs][sampled_idx])
            diff_key = np.random.choice(
                list(filter(lambda s: s != hs, all_states)))
            sampled_idx = np.random.choice(
                list(range(len(self.hash_states2tjs[diff_key]))))
            diff_states.append(self.hash_states2tjs[diff_key][sampled_idx])
        tj_same, tj_same_len = self.tjs.fetch_batch_states(
            list(map(lambda x: x[0], same_states)),
            list(map(lambda x: x[1], same_states))
        )
        tj_diff, tj_diff_len = self.tjs.fetch_batch_states(
            list(map(lambda x: x[0], diff_states)),
            list(map(lambda x: x[1], diff_states))
        )

        src = np.concatenate([tj, tj], axis=0)
        src_len = np.concatenate([tj_len, tj_len], axis=0)
        src2 = np.concatenate([tj_same, tj_diff], axis=0)
        src2_len = np.concatenate([tj_same_len, tj_diff_len], axis=0)

        labels = np.concatenate(
            [np.zeros_like(tj_len), np.ones_like(tj_len)], axis=0)
        return src, src_len, src2, src2_len, labels

    def save_train_pairs(self, t, src, src_len, src2, src2_len, labels):
        src_str = []
        for s in src:
            src_str.append(" ".join(
                map(lambda i: self.tokens[i],
                    filter(lambda x: x != self.hp.padding_val_id, s))
            ))
        src2_str = []
        for s in src2:
            src2_str.append(" ".join(
                map(lambda i: self.tokens[i],
                    filter(lambda x: x != self.hp.padding_val_id, s))
            ))
        np.savez(
            "{}/{}-{}.npz".format(self.model_dir, "train-pairs", t),
            src=src_str, src2=src2_str, src_len=src_len, src2_len=src2_len,
            labels=labels)

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
        next_action_mask = [m[0].next_action_mask for m in b_memory]

        action_mask_t = self.from_bytes(action_mask)
        action_mask_t1 = self.from_bytes(next_action_mask)

        p_states, s_states, p_len, s_len = \
            self.tjs.fetch_batch_states_pair(trajectory_id, state_id)

        # make sure the p_states and s_states are in the same game.
        # otherwise, it won't make sense to use the same action matrix.
        action_matrix = (
            [self.action_collector.get_action_matrix(gid) for gid in game_id])
        action_len = (
            [self.action_collector.get_action_len(gid) for gid in game_id])

        t2 = ctime()
        s_q_actions_target = target_sess.run(
            self.target_model.q_actions,
            feed_dict={self.target_model.src_: s_states,
                       self.target_model.src_len_: s_len,
                       self.target_model.actions_: action_matrix,
                       self.target_model.actions_len_: action_len,
                       self.target_model.actions_mask_: action_mask_t1})

        s_q_actions_dqn = sess.run(
            self.model.q_actions,
            feed_dict={self.model.src_: s_states,
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

        src, src_len, src2, src2_len, labels = self.get_train_pairs(
            trajectory_id, state_id)
        if t % self.hp.save_gap_t == 0:
            self.save_train_pairs(t, src, src_len, src2, src2_len, labels)

        t3 = ctime()
        _, summaries, weighted_loss, abs_loss = sess.run(
            [self.model.merged_train_op, self.model.train_summary_op,
             self.model.weighted_loss,
             self.model.abs_loss],
            feed_dict={self.model.src_: p_states,
                       self.model.src_len_: p_len,
                       self.model.b_weight_: b_weight,
                       self.model.action_idx_: action_id,
                       self.model.actions_mask_: action_mask_t,
                       self.model.expected_q_: expected_q,
                       self.model.actions_: action_matrix,
                       self.model.actions_len_: action_len,
                       self.model.snn_src_: src,
                       self.model.snn_src_len_: src_len,
                       self.model.snn_src2_: src2,
                       self.model.snn_src2_len_: src2_len,
                       self.model.labels_: labels
                       })
        t3_end = ctime()

        self.memo.batch_update(b_idx, abs_loss)
        # self.info('loss: {}'.format(weighted_loss))
        # self.debug('t1: {}, t2: {}, t3: {}'.format(
        #     t1_end - t1, t2_end - t2, t3_end - t3))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)

    def eval_snn(self):
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)
        trajectory_id = [m[0].tid for m in b_memory]
        state_id = [m[0].sid for m in b_memory]
        src, src_len, src2, src2_len, labels = self.get_train_pairs(
            trajectory_id, state_id)
        pred, diff_two_states = self.sess.run(
            [self.model.pred, self.model.diff_two_states],
            feed_dict={self.model.snn_src_: src,
                       self.model.snn_src2_: src2,
                       self.model.snn_src_len_: src_len,
                       self.model.snn_src2_len_: src2_len})

        np.set_printoptions(precision=3, suppress=True, threshold=np.inf)
        self.debug("prediction:\n{}".format(pred))
        self.debug("diff_two_states:\n{}".format(diff_two_states))
