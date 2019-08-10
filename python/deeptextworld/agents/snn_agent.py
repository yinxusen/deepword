import numpy as np
from bert.tokenization import FullTokenizer

from deeptextworld import snn_model
from deeptextworld.agents.base_agent import BaseAgent, ActionDesc, \
    ACT_TYPE_RND_CHOOSE, ACT_TYPE_NN
from deeptextworld.dqn_func import get_random_1Daction, get_best_1Daction, \
    get_best_1D_q
from deeptextworld.hparams import copy_hparams
from deeptextworld.utils import ctime, load_vocab, get_token2idx


class SNNAgent(BaseAgent):
    """
    """
    def __init__(self, hp, model_dir):
        super(SNNAgent, self).__init__(hp, model_dir)

    def get_an_eps_action(self, action_mask):
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param action_mask:
        """
        action_mask = self.from_bytes([action_mask])[0]
        state_text, len_state_text = self.stc.fetch_last_state()
        hs = self.get_hash(state_text)
        if hs not in self.hash_states2tjs:
            self.hash_states2tjs[hs] = []
        self.hash_states2tjs[hs].append(
            (self.tjs.get_current_tid(), self.tjs.get_last_sid()))
        self.eps = 1.0
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
        model_creator = getattr(snn_model, self.hp.model_creator)
        model = snn_model.create_train_model(model_creator, self.hp)
        return model

    def create_eval_model_instance(self):
        model_creator = getattr(snn_model, self.hp.model_creator)
        model = snn_model.create_eval_model(model_creator, self.hp)
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
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)
        trajectory_id = [m[0].tid for m in b_memory]
        state_id = [m[0].sid for m in b_memory]
        src, src_len, src2, src2_len, labels = self.get_train_pairs(
            trajectory_id, state_id)
        if t % self.hp.save_gap_t == 0:
            self.save_train_pairs(t, src, src_len, src2, src2_len, labels)
        _, summaries, loss_eval = sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss],
            feed_dict={self.model.src_: src,
                       self.model.src_len_: src_len,
                       self.model.src2_: src2,
                       self.model.src2_len_: src2_len,
                       self.model.labels_: labels})

        # self.info('loss: {}'.format(loss_eval))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)

    def eval_snn(self):
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)
        trajectory_id = [m[0].tid for m in b_memory]
        state_id = [m[0].sid for m in b_memory]
        src, src_len, src2, src2_len, labels = self.get_train_pairs(
            trajectory_id, state_id)
        pred = self.sess.run(
            self.model.pred,
            feed_dict={self.model.src_: src,
                       self.model.src2_: src2,
                       self.model.src_len_: src_len,
                       self.model.src2_len_: src2_len})
        print("prediction: {}".format(pred))
