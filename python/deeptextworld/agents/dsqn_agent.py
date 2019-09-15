from os import remove as prm

import numpy as np
from numpy.random import choice as npc
from tqdm import trange

from deeptextworld import dsqn_model
from deeptextworld.agents.base_agent import ActionDesc, \
    ACT_TYPE_RND_CHOOSE, ACT_TYPE_NN
from deeptextworld.agents.dqn_agent import TabularDQNAgent
from deeptextworld.dqn_func import get_random_1Daction, get_best_1Daction, \
    get_best_1D_q
from deeptextworld.utils import ctime


class DSQNAgent(TabularDQNAgent):
    """
    """
    def __init__(self, hp, model_dir):
        super(DSQNAgent, self).__init__(hp, model_dir)
        self.hs2tj_prefix = "hs2tj"
        self.hash_states2tjs = {}  # map states to tjs

    def _load_context_objs(self):
        # load others
        super(DSQNAgent, self)._load_context_objs()
        # load hs2tj
        hs2tj_path = self._get_context_obj_path(self.hs2tj_prefix)
        try:
            hs2tj = np.load(hs2tj_path)
            self.hash_states2tjs = hs2tj["hs2tj"][0]
            self.debug("load hash_states2tjs from file")
        except IOError as e:
            self.debug("load hash_states2tjs error:\n{}".format(e))

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
                action_desc.action_type == ACT_TYPE_NN):
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

    def _clean_stale_context(self, tid):
        super(DSQNAgent, self)._clean_stale_context(tid)
        cnt_trashed = 0
        empty_keys = []
        for k in self.hash_states2tjs.keys():
            trashed = self.hash_states2tjs[k].pop(tid, None)
            if trashed is not None:
                cnt_trashed += len(trashed)
            if not self.hash_states2tjs[k]:  # delete the dict if empty
                empty_keys.append(k)
        self.debug("hs2tj deletes {} items".format(cnt_trashed))
        for k in empty_keys:
            self.hash_states2tjs.pop(k, None)
        self.debug("hs2tj deletes {} keys".format(len(empty_keys)))

    def collect_new_sample(self, cleaned_obs, instant_reward, dones, infos):
        actions, all_actions, actions_mask, instant_reward = super(
            DSQNAgent, self).collect_new_sample(
            cleaned_obs, instant_reward, dones, infos)

        if not dones[0]:
            hs, _ = self.stc.fetch_last_state()
            if hs not in self.hash_states2tjs:
                self.hash_states2tjs[hs] = {}
            last_tid = self.tjs.get_current_tid()
            last_sid = self.tjs.get_last_sid()
            if last_tid not in self.hash_states2tjs[hs]:
                self.hash_states2tjs[hs][last_tid] = []
            self.hash_states2tjs[hs][last_tid].append(last_sid)
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

    def get_snn_pairs(self, n):
        non_empty_keys = list(filter(lambda x: self.hash_states2tjs[x],
                                     self.hash_states2tjs.keys()))
        hs_keys = npc(non_empty_keys, size=n)
        diff_keys = [
            npc(list(filter(lambda x: x != k, non_empty_keys)), size=None)
            for k in hs_keys]

        target_tids = []
        same_tids = []
        for k in hs_keys:
            try:
                tid_pair = npc(
                    list(self.hash_states2tjs[k].keys()), size=2, replace=False)
            except ValueError:
                tid_pair = list(self.hash_states2tjs[k].keys()) * 2

            target_tids.append(tid_pair[0])
            same_tids.append(tid_pair[1])

        diff_tids = [npc(list(self.hash_states2tjs[k])) for k in diff_keys]

        target_sids = [npc(list(self.hash_states2tjs[k][tid]))
                       for k, tid in zip(hs_keys, target_tids)]
        same_sids = [npc(list(self.hash_states2tjs[k][tid]))
                     for k, tid in zip(hs_keys, same_tids)]
        diff_sids = [npc(list(self.hash_states2tjs[k][tid]))
                     for k, tid in zip(diff_keys, diff_tids)]

        target_src, target_src_len = self.tjs.fetch_batch_states(
            target_tids, target_sids)
        same_src, same_src_len = self.tjs.fetch_batch_states(
            same_tids, same_sids)
        diff_src, diff_src_len = self.tjs.fetch_batch_states(
            diff_tids, diff_sids)
        src = np.concatenate([target_src, target_src], axis=0)
        src_len = np.concatenate([target_src_len, target_src_len], axis=0)
        src2 = np.concatenate([same_src, diff_src], axis=0)
        src2_len = np.concatenate([same_src_len, diff_src_len], axis=0)
        labels = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
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
        action_matrix = (
            [self.action_collector.get_action_matrix(gid) for gid in game_id])
        action_len = (
            [self.action_collector.get_action_len(gid) for gid in game_id])

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

        t_snn = ctime()
        src, src_len, src2, src2_len, labels = self.get_snn_pairs(
            self.hp.batch_size)
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
        self.debug('t1: {}, t2: {}, t3: {}, t_snn_mk_pairs: {}'.format(
            t1_end - t1, t2_end - t2, t3_end - t3, t_snn_end - t_snn))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)

    def eval_snn(self, eval_data_size):
        src, src_len, src2, src2_len, labels = self.get_snn_pairs(
            eval_data_size)
        labels = labels.astype(np.int32)
        pred, diff_two_states = self.sess.run(
            [self.model.pred, self.model.diff_two_states],
            feed_dict={self.model.snn_src_: src,
                       self.model.snn_src2_: src2,
                       self.model.snn_src_len_: src_len,
                       self.model.snn_src2_len_: src2_len})
        pred_labels = (pred > 0.5).astype(np.int32)
        accuracy = np.mean(np.equal(labels, pred_labels))
        return accuracy


class DSQNAlterAgent(DSQNAgent):
    """
    Train DRRN and SNN alternatively.
    """
    def _train_snn(self, sess, n_iters, summary_writer, t):
        for i in trange(n_iters):
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

    def train_impl(self, sess, t, summary_writer, target_sess, target_model):
        if self.time_to_save():
            self.debug("training SNN")
            n_iters = self.hp.save_gap_t // 10
            t_snn = ctime()
            # train SNN
            self._train_snn(sess, n_iters, summary_writer, t)
            t_snn_end = ctime()
            self.debug("training DQN")
        else:
            t_snn = 0
            t_snn_end = 0
            n_iters = 0
            pass

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
        # self.info('loss: {}'.format(weighted_loss))
        self.debug('t1: {}, t2: {}, t3: {}, t_snn_training: {} / {} (iters)'.format(
            t1_end - t1, t2_end - t2, t3_end - t3, t_snn_end - t_snn, n_iters))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)