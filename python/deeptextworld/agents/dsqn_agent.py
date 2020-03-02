from bisect import bisect_left
from multiprocessing.pool import ThreadPool
from os import remove as prm
from os.path import join as pjoin

import numpy as np
import tensorflow as tf
from numpy.random import choice as npc

from deeptextworld.agents.base_agent import ActionDesc, ACT_TYPE
from deeptextworld.agents.dqn_agent import TabularDQNAgent
from deeptextworld.models import dsqn_model
from deeptextworld.models.dqn_func import get_random_1Daction, \
    get_best_1Daction, get_best_1D_q
from deeptextworld.utils import ctime


class DSQNAgent(TabularDQNAgent):
    def __init__(self, hp, model_dir):
        super(DSQNAgent, self).__init__(hp, model_dir)
        self.hs2tj_prefix = "hs2tj"
        self.hash_states2tjs = {}  # map states to tjs
        self.pool_train = ThreadPool(processes=2)

    def init_hs2tj(self, hs2tj_path, with_loading=True):
        hash_states2tjs = {}
        if with_loading:
            try:
                hs2tj = np.load(hs2tj_path, allow_pickle=True)
                hash_states2tjs = hs2tj["hs2tj"][0]
                self.debug("load hash_states2tjs from file")
            except IOError as e:
                self.debug("load hash_states2tjs error:\n{}".format(e))
        return hash_states2tjs

    def _load_context_objs(self):
        # load others
        super(DSQNAgent, self)._load_context_objs()
        # load hs2tj
        hs2tj_path = self._get_context_obj_path(self.hs2tj_prefix)
        self.hash_states2tjs = self.init_hs2tj(
            hs2tj_path, with_loading=self.is_training)

    def _save_context_objs(self):
        super(DSQNAgent, self)._save_context_objs()
        hs2tj_path = self._get_context_obj_new_path(self.hs2tj_prefix)
        np.savez(hs2tj_path, hs2tj=[self.hash_states2tjs])

    def get_compatible_snapshot_tag(self):
        # get parent valid tags
        valid_tags = super(DSQNAgent, self).get_compatible_snapshot_tag()
        valid_tags = set(valid_tags)
        # mix valid tags w/ context objs
        hs2tj_tags = self.get_path_tags(self.model_dir, self.hs2tj_prefix)
        valid_tags.intersection_update(hs2tj_tags)
        return list(valid_tags)

    def _delete_stale_context_objs(self):
        super(DSQNAgent, self)._delete_stale_context_objs()
        if self._stale_tags is not None:
            for tag in self._stale_tags:
                prm(self._get_context_obj_path_w_tag(self.hs2tj_prefix, tag))

    def _jitter_go_condition(self, action_desc, admissible_go_actions):
        if not (self.hp.jitter_go and
                action_desc.action in admissible_go_actions and
                action_desc.action_type == ACT_TYPE.policy_drrn):
            return False
        else:
            if self.is_training:
                return np.random.random() > 1. - self.hp.jitter_train_prob
            else:
                return np.random.random() > 1. - self.hp.jitter_eval_prob

    def choose_action(
            self, actions, all_actions, actions_mask, instant_reward):
        """
        Choose an action by
          1) try rule-based policy;
          2) try epsilon search learned policy;
          3) jitter go
        :param actions:
        :param all_actions:
        :param actions_mask:
        :param instant_reward:
        :return:
        """
        action_desc = self.rule_based_policy(
            actions, all_actions, instant_reward)
        if action_desc.action_idx is None:
            action_desc = self.random_walk_for_collecting_fp(
                actions, all_actions)
            if action_desc.action_idx is None:
                action_desc = self.get_an_eps_action(actions_mask)
                action_desc = self.jitter_go_action(
                    action_desc, actions, all_actions)
            else:
                pass
        else:
            pass
        return action_desc

    def _clean_stale_context(self, tids):
        super(DSQNAgent, self)._clean_stale_context(tids)
        if not tids:
            return
        hs2tj_cleaned = {}
        for k in self.hash_states2tjs.keys():
            start_t = bisect_left(
                [t for t, s in self.hash_states2tjs[k]], max(tids))
            if not self.hash_states2tjs[k][start_t:]:
                hs2tj_cleaned[k] = self.hash_states2tjs[k][start_t:]
            else:
                self.debug("remove key {} from hs2tj".format(k))
        self.hash_states2tjs = hs2tj_cleaned

    def collect_new_sample(self, cleaned_obs, instant_reward, dones, infos):
        actions, all_actions, actions_mask, instant_reward = super(
            DSQNAgent, self).collect_new_sample(
            cleaned_obs, instant_reward, dones, infos)

        if not dones[0]:
            hs, _ = self.stc.fetch_last_state()
            if hs not in self.hash_states2tjs:
                self.hash_states2tjs[hs] = []
            last_tid = self.tjs.get_current_tid()
            last_sid = self.tjs.get_last_sid()
            self.hash_states2tjs[hs].append((last_tid, last_sid))
        else:
            pass  # final states are not considered

        return actions, all_actions, actions_mask, instant_reward

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
            q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
                self.model.src_: [indexed_state_t],
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
        model_creator = getattr(dsqn_model, self.hp.model_creator)
        model = dsqn_model.create_train_model(
            model_creator, self.hp, device_placement=device)
        return model

    def create_eval_model_instance(self, device):
        model_creator = getattr(dsqn_model, self.hp.model_creator)
        model = dsqn_model.create_eval_model(
            model_creator, self.hp, device_placement=device)
        return model

    def get_snn_pairs(self, n):
        # target key set should contain items more than twice, since we need to
        # separate target set and same set.
        target_key_set = list(
            filter(lambda x: len(self.hash_states2tjs[x]) >= 2,
                   self.hash_states2tjs.keys()))
        self.debug(
            "choose from {} keys for SNN target".format(len(target_key_set)))
        hs_keys = npc(target_key_set, size=n)

        diff_keys_duo = npc(
            list(self.hash_states2tjs.keys()), replace=False, size=(n, 2))
        diff_keys = diff_keys_duo[:, 0]
        same_key_ids = np.where(hs_keys == diff_keys)[0]
        diff_keys[same_key_ids] = diff_keys_duo[same_key_ids, 1]

        tgt_set = []
        same_set = []
        diff_set = []
        for hk, dk in zip(hs_keys, diff_keys):
            samples_ids = npc(
                len(self.hash_states2tjs[hk]), size=2, replace=False)
            tgt_set.append(self.hash_states2tjs[hk][samples_ids[0]])
            same_set.append(self.hash_states2tjs[hk][samples_ids[1]])
            diff_set.append(
                self.hash_states2tjs[dk][npc(len(self.hash_states2tjs[dk]))])

        tgt_src, tgt_src_len = self.tjs.fetch_batch_states_impl(tgt_set)
        same_src, same_src_len = self.tjs.fetch_batch_states_impl(same_set)
        diff_src, diff_src_len = self.tjs.fetch_batch_states_impl(diff_set)

        src = np.concatenate([tgt_src, tgt_src], axis=0)
        src_len = np.concatenate([tgt_src_len, tgt_src_len], axis=0)
        src2 = np.concatenate([same_src, diff_src], axis=0)
        src2_len = np.concatenate([same_src_len, diff_src_len], axis=0)
        labels = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
        return src, src_len, src2, src2_len, labels

    def save_train_pairs(self, t, src, src_len, src2, src2_len, labels):
        src_str = []
        for s in src:
            src_str.append(" ".join(
                map(lambda i: self.tokenizer.inv_vocab[i],
                    filter(lambda x: x != self.hp.padding_val_id, s))
            ))
        src2_str = []
        for s in src2:
            src2_str.append(" ".join(
                map(lambda i: self.tokenizer.inv_vocab[i],
                    filter(lambda x: x != self.hp.padding_val_id, s))
            ))
        np.savez(
            "{}/{}-{}.npz".format(self.model_dir, "train-pairs", t),
            src=src_str, src2=src2_str, src_len=src_len, src2_len=src2_len,
            labels=labels)

    def train_impl(self, sess, t, summary_writer, target_sess, target_model):

        def get_snn_data(batch_size):
            return self.get_snn_pairs(batch_size)

        async_snn_data = self.pool_train.apply_async(
            get_snn_data, args=(self.hp.batch_size,))

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
            [self.actor.get_action_matrix(gid) for gid in game_id])
        action_len = (
            [self.actor.get_action_len(gid) for gid in game_id])

        def run_target_net():
            q_actions = target_sess.run(
                target_model.q_actions,
                feed_dict={target_model.src_: s_states,
                           target_model.src_len_: s_len,
                           target_model.actions_: action_matrix,
                           target_model.actions_len_: action_len,
                           target_model.actions_mask_: action_mask_t1})
            return q_actions

        t2 = ctime()
        async_q_actions = self.pool_train.apply_async(run_target_net)

        s_q_actions_dqn = sess.run(
            self.model.q_actions,
            feed_dict={self.model.src_: s_states,
                       self.model.src_len_: s_len,
                       self.model.actions_: action_matrix,
                       self.model.actions_len_: action_len,
                       self.model.actions_mask_: action_mask_t1})
        s_q_actions_target = async_q_actions.get()
        t2_end = ctime()

        expected_q = np.zeros_like(reward)
        for i in range(len(expected_q)):
            expected_q[i] = reward[i]
            if not is_terminal[i]:
                s_argmax_q, _ = get_best_1D_q(
                    s_q_actions_dqn[i, :], mask=action_mask_t1[i])
                expected_q[i] += (
                    self.hp.final_gamma * s_q_actions_target[i, s_argmax_q])

        t_snn = ctime()
        src, src_len, src2, src2_len, labels = async_snn_data.get()
        t_snn_end = ctime()

        t3 = ctime()
        _, summaries, weighted_loss, abs_loss = sess.run(
            [self.model.merged_train_op, self.model.weighted_train_summary_op,
             self.model.weighted_loss, self.model.abs_loss],
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
        if t % 1000 == 0:
            self.debug(
                't: {}, t1: {}, t2: {}, t3: {}, t_snn_mk_pairs: {}'.format(
                    t, t1_end - t1, t2_end - t2, t3_end - t3,
                    t_snn_end - t_snn))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)

    def eval_snn(self, eval_data_size, batch_size=32):
        self.info("start eval with size {}".format(eval_data_size))
        n_iter = (eval_data_size // batch_size) + 1
        total_acc = 0
        total_samples = 0
        for i in range(n_iter):
            self.debug("eval snn iter {} total {}".format(i, n_iter))
            src, src_len, src2, src2_len, labels = self.get_snn_pairs(
                batch_size)
            non_empty_src = list(filter(
                lambda x: x[1][0] != 0 and x[1][1] != 0,
                enumerate(zip(src_len, src2_len))))
            non_empty_src_idx = [x[0] for x in non_empty_src]
            src = src[non_empty_src_idx, :]
            src_len = src_len[non_empty_src_idx]
            src2 = src2[non_empty_src_idx, :]
            src2_len = src2_len[non_empty_src_idx]
            labels = labels[non_empty_src_idx]
            labels = labels.astype(np.int32)
            pred, diff_two_states = self.sess.run(
                [self.model.pred, self.model.diff_two_states],
                feed_dict={self.model.snn_src_: src,
                           self.model.snn_src2_: src2,
                           self.model.snn_src_len_: src_len,
                           self.model.snn_src2_len_: src2_len})
            pred_labels = (pred > 0).astype(np.int32)
            total_acc += np.sum(np.equal(labels, pred_labels))
            total_samples += len(src)
        if total_samples == 0:
            avg_acc = -1
        else:
            avg_acc = total_acc * 1. / total_samples
            self.debug("valid sample size {}".format(total_samples))
        return avg_acc


class DSQNAlterAgent(DSQNAgent):
    """
    Train DRRN and SNN alternatively.
    """
    def __init__(self, hp, model_dir):
        super(DSQNAlterAgent, self).__init__(hp, model_dir)
        self.start_snn_training = True

    def _train_snn(self, sess, n_iters, summary_writer, t):
        for i in range(n_iters):
            self.debug("training SNN: {}/{} epochs".format(i, n_iters))
            src, src_len, src2, src2_len, labels = self.get_snn_pairs(
                self.hp.batch_size)
            _, summaries, snn_loss = sess.run(
                [self.model.snn_train_op, self.model.snn_train_summary_op,
                 self.model.snn_loss],
                feed_dict={self.model.snn_src_: src,
                           self.model.snn_src_len_: src_len,
                           self.model.snn_src2_: src2,
                           self.model.snn_src2_len_: src2_len,
                           self.model.labels_: labels
                           })
            summary_writer.add_summary(summaries, t - self.hp.observation_t + i)

    def train_snn(self, sess, summary_writer, t):
        """
        Training sequences for DRRN and SNN:

            SNN -- n_iters epochs
            DRRN -- save_gap_t epochs
            save model --- time to save
            copy model as target
            SNN -- n_iters epochs
            DRRN
            ...

        In this way, the target model won't be contaminated
        by the SNN training.
        :param sess:
        :param summary_writer:
        :param t:
        :return:
        """
        if self.start_snn_training:
            n_iters = self.hp.snn_train_epochs
            t_snn = ctime()
            self._train_snn(sess, n_iters, summary_writer, t)
            t_snn_end = ctime()
            # will set to True after save_snapshot
            self.start_snn_training = False
            self.debug(
                "t-snn: {} / {} iters".format(t_snn_end - t_snn, n_iters))

    def save_snapshot(self):
        super(DSQNAlterAgent, self).save_snapshot()
        self.start_snn_training = True

    def train_one_batch(self):
        if self.total_t == self.hp.observation_t:
            self.epoch_start_t = ctime()
        self.train_snn(self.sess, self.train_summary_writer, self.total_t)
        # if there is not a well-trained model, it is unreasonable
        # to use target model.
        self.train_impl(
            self.sess, self.total_t, self.train_summary_writer,
            self.target_sess if self.target_sess else self.sess,
            self.target_model if self.target_model else self.model)
        self._save_agent_n_reload_target()

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

        p_states, s_states, p_len, s_len = \
            self.tjs.fetch_batch_states_pair(trajectory_id, state_id)

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
                       target_model.src_len_: s_len,
                       target_model.actions_: action_matrix,
                       target_model.actions_len_: action_len,
                       target_model.actions_mask_: action_mask_t1})

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

        t3 = ctime()
        _,  summaries, weighted_loss, abs_loss = sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={self.model.src_: p_states,
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
            self.debug(
                't: {}, t1: {}, t2: {}, t3: {}'.format(
                    t, t1_end - t1, t2_end - t2, t3_end - t3))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)


class BertDSQNAgent(DSQNAlterAgent):
    pass


class BertDSQNIndAgent(DSQNAlterAgent):
    """Bert DSQN Independent Agent, means we split SNN and DRRN"""
    def __init__(self, hp, model_dir):
        super(BertDSQNIndAgent, self).__init__(hp, model_dir)
        self.snn_model = None
        self.snn_sess = None
        self.snn_saver = None
        self.snn_ckpt_path = pjoin(self.model_dir, 'snn_last_weights')
        self.snn_ckpt_prefix = pjoin(self.snn_ckpt_path, 'after-epoch')
        self.snn_best_ckpt_path = pjoin(self.model_dir, 'snn_best_weights')
        self.snn_best_ckpt_prefix = pjoin(
            self.snn_best_ckpt_path, 'after-epoch')
        self.train_snn_summary_writer = None
        self.snn_bert_loader = None
        self.drrn_bert_loader = None

    def create_model_instance(self, device):
        model_creator = getattr(dsqn_model, self.hp.model_creator)
        model = dsqn_model.create_train_drrn_model(
            model_creator, self.hp, device_placement=device)
        return model

    def create_eval_model_instance(self, device):
        model_creator = getattr(dsqn_model, self.hp.model_creator)
        model = dsqn_model.create_eval_drrn_model(
            model_creator, self.hp, device_placement=device)
        return model

    def create_snn_model_instance(self, device):
        model_creator = getattr(dsqn_model, self.hp.model_creator)
        model = dsqn_model.create_train_snn_model(
            model_creator, self.hp, device_placement=device)
        return model

    def create_snn_eval_model_instance(self, device):
        model_creator = getattr(dsqn_model, self.hp.model_creator)
        model = dsqn_model.create_eval_snn_model(
            model_creator, self.hp, device_placement=device)
        return model

    def train_snn(self, sess, summary_writer, t):
        """
        Training sequences for DRRN and SNN:

            SNN -- n_iters epochs
            DRRN -- save_gap_t epochs
            save model --- time to save
            copy model as target
            SNN -- n_iters epochs
            DRRN
            ...

        In this way, the target model won't be contaminated
        by the SNN training.
        :param sess:
        :param summary_writer:
        :param t:
        :return:
        """
        if self.start_snn_training:
            try:
                self.snn_bert_loader.restore(
                    self.snn_sess, tf.train.latest_checkpoint(self.ckpt_path))
                self.debug("bert from DRRN to SNN")
            except ValueError as e:
                self.debug(e)
            n_iters = self.hp.snn_train_epochs
            t_snn = ctime()
            self._train_snn(sess, n_iters, summary_writer, t)
            t_snn_end = ctime()
            # will set to True after save_snapshot
            self.start_snn_training = False
            self.debug(
                "t-snn: {} / {} iters".format(t_snn_end - t_snn, n_iters))
            self.snn_saver.save(
                self.snn_sess, self.snn_ckpt_prefix,
                global_step=tf.train.get_or_create_global_step(
                    graph=self.snn_model.graph))
            self.drrn_bert_loader.restore(
                self.sess, tf.train.latest_checkpoint(self.snn_ckpt_path))
            self.debug("bert from SNN to DRRN")

    def _train_snn(self, sess, n_iters, summary_writer, t):
        for i in range(n_iters):
            self.debug("training SNN: {}/{} epochs".format(i, n_iters))
            src, src_len, src2, src2_len, labels = self.get_snn_pairs(
                self.hp.batch_size // 2)
            _, summaries, snn_loss = sess.run(
                [self.snn_model.snn_train_op,
                 self.snn_model.snn_train_summary_op,
                 self.snn_model.snn_loss],
                feed_dict={self.snn_model.snn_src_: src,
                           self.snn_model.snn_src_len_: src_len,
                           self.snn_model.snn_src2_: src2,
                           self.snn_model.snn_src2_len_: src2_len,
                           self.snn_model.labels_: labels
                           })
            summary_writer.add_summary(summaries, t - self.hp.observation_t + i)

    def create_n_load_snn_model(self, load_best=False, is_training=True):
        start_t = 0
        if is_training:
            model = self.create_snn_model_instance(device="/device:GPU:0")
            self.info("create snn train model")
        else:
            model = self.create_snn_eval_model_instance(device="/device:GPU:1")
            self.info("create snn eval model")

        conf = tf.ConfigProto(
            log_device_placement=True, allow_soft_placement=True)
        sess = tf.Session(graph=model.graph, config=conf)
        with model.graph.as_default():
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(
                max_to_keep=self.hp.max_snapshot_to_keep,
                save_relative_paths=True)
            if load_best:
                restore_from = tf.train.latest_checkpoint(
                    self.snn_best_ckpt_path)
            else:
                restore_from = tf.train.latest_checkpoint(self.snn_ckpt_path)

            if restore_from is not None:
                # Reload weights from directory if specified
                self.info(
                    "Try to restore parameters from: {}".format(restore_from))
                try:
                    saver.restore(sess, restore_from)
                except Exception as e:
                    self.debug("Restoring failed: {}".format(e))
                    all_saved_vars = list(
                        map(lambda v: v[0],
                            tf.train.list_variables(restore_from)))
                    self.debug(
                        "Try to restore with safe saver with vars: {}".format(
                            "\n".join(all_saved_vars)))
                    safe_saver = tf.train.Saver(var_list=all_saved_vars)
                    safe_saver.restore(sess, restore_from)
                if not self.hp.start_t_ignore_model_t:
                    global_step = tf.train.get_or_create_global_step()
                    trained_steps = sess.run(global_step)
                    start_t = trained_steps + self.hp.observation_t
            else:
                self.info('No checkpoint to load, training from scratch')
        return sess, start_t, saver, model

    def _init_impl(self, load_best=False, restore_from=None):
        super(BertDSQNIndAgent, self)._init_impl(load_best, restore_from)
        (self.snn_sess, _, self.snn_saver, self.snn_model
         ) = self.create_n_load_snn_model(load_best, self.is_training)
        if self.is_training:
            train_snn_summary_dir = pjoin(
                self.model_dir, "snn-summaries", "train")
            self.train_snn_summary_writer = tf.summary.FileWriter(
                train_snn_summary_dir, self.snn_sess.graph)
            with self.snn_model.graph.as_default():
                snn_all_trainable = tf.trainable_variables()
            bert_vars = list(
                filter(lambda v: "bert-state-encoder/bert" in v.name,
                       snn_all_trainable))
            self.debug("Bert vars: {}".format(bert_vars))
            self.snn_bert_loader = tf.train.Saver(var_list=bert_vars)
            with self.model.graph.as_default():
                drrn_all_trainable = tf.trainable_variables()
            bert_vars = list(
                filter(lambda v: "bert-state-encoder/bert" in v.name,
                       drrn_all_trainable))
            self.debug("Bert vars: {}".format(bert_vars))
            self.drrn_bert_loader = tf.train.Saver(var_list=bert_vars)
        else:
            pass

    def train_one_batch(self):
        if self.total_t == self.hp.observation_t:
            self.epoch_start_t = ctime()
        self.train_snn(
            self.snn_sess, self.train_snn_summary_writer, self.total_t)
        self.train_impl(
            self.sess, self.total_t, self.train_summary_writer,
            self.target_sess if self.target_sess else self.sess,
            self.target_model if self.target_model else self.model)
        self._save_agent_n_reload_target()

    def eval_snn(self, eval_data_size, batch_size=16):
        self.info("start eval with size {}".format(eval_data_size))
        n_iter = (eval_data_size // batch_size) + 1
        total_acc = 0
        total_samples = 0
        for i in range(n_iter):
            src, src_len, src2, src2_len, labels = self.get_snn_pairs(
                batch_size)
            non_empty_src = list(filter(
                lambda x: x[1][0] != 0 and x[1][1] != 0,
                enumerate(zip(src_len, src2_len))))
            non_empty_src_idx = [x[0] for x in non_empty_src]
            src = src[non_empty_src_idx, :]
            src_len = src_len[non_empty_src_idx]
            src2 = src2[non_empty_src_idx, :]
            src2_len = src2_len[non_empty_src_idx]
            labels = labels[non_empty_src_idx]
            labels = labels.astype(np.int32)
            pred, diff_two_states = self.snn_sess.run(
                [self.snn_model.pred, self.snn_model.diff_two_states],
                feed_dict={self.snn_model.snn_src_: src,
                           self.snn_model.snn_src2_: src2,
                           self.snn_model.snn_src_len_: src_len,
                           self.snn_model.snn_src2_len_: src2_len})
            pred_labels = (pred > 0).astype(np.int32)
            total_acc += np.sum(np.equal(labels, pred_labels))
            total_samples += len(src)
        if total_samples == 0:
            avg_acc = -1
        else:
            avg_acc = total_acc * 1. / total_samples
            self.debug("valid sample size {}".format(total_samples))
        return avg_acc
