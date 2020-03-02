from copy import deepcopy
from os import remove as prm

import numpy as np
from textworld import EnvInfos

from deeptextworld.agents.base_agent import BaseAgent, ActionDesc, ACT_TYPE, \
    INFO_KEY
from deeptextworld.models import dqn_model
from deeptextworld.models.dqn_func import get_best_2D_q
from deeptextworld.models.dqn_func import get_random_1Daction, \
    get_best_1Daction, \
    get_best_1D_q
from deeptextworld.trajectory import StateTextCompanion
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
            won=True,
            admissible_commands=True)

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
                action=action)
        else:
            indexed_state_t, lens_t = self.tjs.fetch_last_state()
            q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
                self.model.src_: [indexed_state_t],
                self.model.src_len_: [lens_t]
            })[0]
            action_idx, q_max, action = get_best_1Daction(
                q_actions_t, self.actor.actions,
                mask=action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE.policy_drrn,
                action_idx=action_idx, action=action)
        return action_desc

    def create_model_instance(self, device):
        model_creator = getattr(dqn_model, self.hp.model_creator)
        model = dqn_model.create_train_model(model_creator, self.hp)
        return model

    def create_eval_model_instance(self, device):
        model_creator = getattr(dqn_model, self.hp.model_creator)
        model = dqn_model.create_eval_model(model_creator, self.hp)
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
        reward = [m[0].reward for m in b_memory]
        is_terminal = [m[0].is_terminal for m in b_memory]
        next_action_mask = [m[0].next_action_mask for m in b_memory]

        action_mask_t1 = self.from_bytes(next_action_mask)

        p_states, s_states, p_len, s_len = \
            self.tjs.fetch_batch_states_pair(trajectory_id, state_id)

        t2 = ctime()
        s_q_actions_target = target_sess.run(
            target_model.q_actions,
            feed_dict={target_model.src_: s_states,
                       target_model.src_len_: s_len})

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
            t1_end - t1, t2_end - t2, t3_end - t3))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)


class TabularDQNAgent(DQNAgent):
    def __init__(self, hp, model_dir):
        super(TabularDQNAgent, self).__init__(hp, model_dir)
        self.q_mat_prefix = "q_mat"
        self.q_mat = {}  # map hash of a state to a q-vec
        self.target_q_mat = {}  # target q-mat for Double DQN
        self.stc_prefix = "state_text"
        self.stc = None

    def init_state_text(self, state_text_path, with_loading=True):
        stc = StateTextCompanion()
        if with_loading:
            try:
                stc.load_tjs(state_text_path)
            except IOError as e:
                self.info("load trajectory error: \n{}".format(e))
        return stc

    def _load_context_objs(self):
        # load others
        super(TabularDQNAgent, self)._load_context_objs()
        # load q_mat
        # TODO: q_mat is actually the model for TabularDQN, but let's make it
        # TODO: easier by loading like a context object.
        q_mat_path = self._get_context_obj_path(self.q_mat_prefix)
        try:
            npz_q_mat = np.load(q_mat_path)
            q_mat_key = npz_q_mat["q_mat_key"]
            q_mat_val = npz_q_mat["q_mat_val"]
            self.q_mat = dict(zip(q_mat_key, q_mat_val))
            self.debug("load q_mat from file")
            self.target_q_mat = deepcopy(self.q_mat)
            self.debug("init target_q_mat with q_mat")
        except IOError as e:
            self.debug("load q_mat error:\n{}".format(e))
        # load stc
        stc_path = self._get_context_obj_path(self.stc_prefix)
        self.stc = self.init_state_text(stc_path, with_loading=True)

    def _save_context_objs(self):
        super(TabularDQNAgent, self)._save_context_objs()
        q_mat_path = self._get_context_obj_new_path(self.q_mat_prefix)
        stc_path = self._get_context_obj_new_path(self.stc_prefix)
        self.stc.save_tjs(stc_path)
        np.savez(
            q_mat_path,
            q_mat_key=list(self.q_mat.keys()),
            q_mat_val=list(self.q_mat.values()))
        self.target_q_mat = deepcopy(self.q_mat)
        self.debug("target q_mat is updated with q_mat")

    def get_compatible_snapshot_tag(self):
        # get parent valid tags
        valid_tags = super(TabularDQNAgent, self).get_compatible_snapshot_tag()
        valid_tags = set(valid_tags)
        # mix valid tags w/ context objs
        q_mat_tags = self.get_path_tags(self.model_dir, self.q_mat_prefix)
        stc_tags = self.get_path_tags(self.model_dir, self.stc_prefix)
        valid_tags.intersection_update(q_mat_tags)
        valid_tags.intersection_update(stc_tags)
        return list(valid_tags)

    def _delete_stale_context_objs(self):
        super(TabularDQNAgent, self)._delete_stale_context_objs()
        if self._stale_tags is not None:
            for tag in self._stale_tags:
                prm(self._get_context_obj_path_w_tag(self.q_mat_prefix, tag))
                prm(self._get_context_obj_path_w_tag(self.stc_prefix, tag))

    def _start_episode_impl(self, obs, infos):
        super(TabularDQNAgent, self)._start_episode_impl(obs, infos)
        self.stc.add_new_tj(tid=self.tjs.get_current_tid())

    def _jitter_go_condition(self, action_desc, admissible_go_actions):
        if not (self.hp.jitter_go and
                action_desc.action in admissible_go_actions and
                action_desc.action_type == ACT_TYPE.policy_tbl):
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
          1) try epsilon search learned policy;
          2) jitter go
        :param actions:
        :param all_actions:
        :param actions_mask:
        :param instant_reward:
        :return:
        """
        action_desc = self.random_walk_for_collecting_fp(
            actions, all_actions)
        if action_desc.action_idx is None:
            action_desc = self.get_an_eps_action(actions_mask)
            action_desc = self.jitter_go_action(
                action_desc, actions, all_actions)
        else:
            pass
        return action_desc

    def _clean_stale_context(self, tids):
        super(TabularDQNAgent, self)._clean_stale_context(tids)
        self.debug("stc deletes {}".format(tids))
        self.stc.request_delete_keys(tids)

    def collect_new_sample(self, cleaned_obs, instant_reward, dones, infos):
        actions, all_actions, actions_mask, instant_reward = super(
            TabularDQNAgent, self).collect_new_sample(
            cleaned_obs, instant_reward, dones, infos)

        # due to game design flaw, we need to make a new terminal
        # observation + inventory
        # because the game terminal observation + inventory is the same with
        # its previous state
        if not dones[0]:
            state_text = (
                infos[INFO_KEY.desc][0] + "\n" + infos[INFO_KEY.inventory][0])
        else:
            state_text = (
                "terminal and win" if infos[INFO_KEY.won]
                else "terminal and lose")
        self.stc.append(self.get_hash(state_text))

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
            hs, _ = self.stc.fetch_last_state()
            q_actions_t = self.q_mat.get(hs, np.zeros(self.hp.n_actions))
            action_idx, q_max, action = get_best_1Daction(
                q_actions_t, self.actor.actions,
                mask=action_mask)
            action_desc = ActionDesc(
                action_type=ACT_TYPE.policy_tbl, action_idx=action_idx,
                token_idx=self.actor.action_matrix[action_idx],
                action_len=self.actor.action_len[action_idx],
                action=action)
        return action_desc

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
        reward = [m[0].reward for m in b_memory]
        is_terminal = [m[0].is_terminal for m in b_memory]
        next_action_mask = [m[0].next_action_mask for m in b_memory]

        action_mask_t1 = self.from_bytes(next_action_mask)

        p_states, s_states, p_len, s_len = \
            self.stc.fetch_batch_states_pair(trajectory_id, state_id)

        t2 = ctime()
        s_q_actions_target = np.asarray(
            [self.target_q_mat.get(s, np.zeros(self.hp.n_actions))
             for s in s_states])
        s_q_actions_dqn = np.asarray(
            [self.q_mat.get(s, np.zeros(self.hp.n_actions))
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
        abs_loss = np.zeros_like(reward)
        for i, ps in enumerate(p_states):
            if ps not in self.q_mat:
                self.q_mat[ps] = np.zeros(self.hp.n_actions)
            prev_q_val = self.q_mat[ps][action_id[i]]
            delta_q_val = expected_q[i] - prev_q_val
            abs_loss[i] = abs(delta_q_val)
            self.q_mat[ps][action_id[i]] = (
                    prev_q_val + delta_q_val * b_weight[i])
        t3_end = ctime()

        self.memo.batch_update(b_idx, abs_loss)

        self.debug('t1: {}, t2: {}, t3: {}'.format(
            t1_end - t1, t2_end - t2, t3_end - t3))


class GenDQNAgent(DQNAgent):
    def __init__(self, hp, model_dir):
        super(GenDQNAgent, self).__init__(hp, model_dir)

    def select_additional_infos(self):
        """
        additional information needed when playing the game
        """
        return EnvInfos(
            description=True,
            inventory=True,
            max_score=True,
            won=True,
            admissible_commands=True)

    @classmethod
    def negative_response_reward(cls, master):
        if master == "that 's not a verb i recognise .":
            return 1
        return 0

    def get_an_eps_action(self, action_mask):
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param action_mask:
        """
        indexed_state_t, lens_t = self.tjs.fetch_last_state()
        beam_size = 1
        temperature = 1
        self.debug("temperature: {}".format(temperature))
        res = self.sess.run(
            [self.model.decoded_idx_infer, self.model.col_eos_idx,
             self.model.decoded_logits_infer, self.model.p_gen_infer],
            feed_dict={
                self.model.src_: [indexed_state_t],
                self.model.src_len_: [lens_t],
                self.model.temperature_: temperature,
                self.model.beam_size_: beam_size,
                self.model.use_greedy_: False
            })
        action_idx = res[0]
        col_eos_idx = res[1]
        decoded_logits = res[2]
        # self.debug("decoded logits: {}".format(decoded_logits))
        p_gen = res[3]

        res_summary = []
        special_tokens = {self.hp.padding_val, self.hp.eos}
        for bid in range(beam_size):
            action = " ".join(
                filter(lambda t: t not in special_tokens,
                       self.tokenizer.convert_ids_to_tokens(
                           action_idx[bid, :col_eos_idx[bid]])))
            res_summary.append(
                (action_idx[bid], col_eos_idx[bid],
                 action, p_gen[bid],
                 np.sum(decoded_logits[bid, :col_eos_idx[bid]])
                 / col_eos_idx[bid]))

        res_summary = list(reversed(sorted(res_summary, key=lambda x: x[-1])))
        top_action = res_summary[0]

        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_gen, action_idx=None,
            token_idx=top_action[0], action_len=top_action[1],
            action=top_action[2])

        self.debug("generated actions:\n{}".format(
            "\n".join(
                [" ".join(
                    map(lambda a_p: "{}[{:.2f}]".format(a_p[0], a_p[1]),
                        zip(ac[2].split(), list(ac[3])))) + "\t{}".format(ac[4])
                 for ac in res_summary])))
        return action_desc

    def get_instant_reward(self, score, master, is_terminal, won):
        """
        increase instance reward 10 times to fit cross entropy loss trained
        model
        """
        ir = super(GenDQNAgent, self).get_instant_reward(
            score, master, is_terminal, won)
        return ir * 10

    def create_model_instance(self, device):
        model_creator = getattr(dqn_model, self.hp.model_creator)
        model = dqn_model.create_train_gen_model(
            model_creator, self.hp, device_placement=device)
        return model

    def create_eval_model_instance(self, device):
        model_creator = getattr(dqn_model, self.hp.model_creator)
        model = dqn_model.create_eval_gen_model(
            model_creator, self.hp, device_placement=device)
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
        at_id = [m[0].token_id for m in b_memory]
        action_len = [m[0].a_len for m in b_memory]
        reward = [m[0].reward for m in b_memory]
        is_terminal = [m[0].is_terminal for m in b_memory]
        at_id_wo_eos = np.asarray(at_id)
        at_id_wo_eos[
            range(len(at_id)), np.asarray(action_len) - 1] = 0
        at_id_in = np.concatenate(
            [np.asarray([[self.hp.sos_id]] * len(action_len)),
             at_id_wo_eos[:, :-1]], axis=1)

        at_id = np.asarray(at_id)
        self.debug("action in/out example:\n{} -- {}\n{} -- {}".format(
            at_id_in[0, :],
            self.tokenizer.convert_ids_to_tokens(
                at_id_in[0, :action_len[0]]),
            at_id[0, :],
            self.tokenizer.convert_ids_to_tokens(
                at_id[0, :action_len[0]])))

        p_states, s_states, p_len, s_len = \
            self.tjs.fetch_batch_states_pair(trajectory_id, state_id)

        t2 = ctime()
        s_q_actions_target = target_sess.run(
            target_model.q_actions,
            feed_dict={target_model.src_: s_states,
                       target_model.src_len_: s_len,
                       target_model.action_idx_: at_id_in})

        s_q_actions_dqn = sess.run(
            self.model.q_actions,
            feed_dict={self.model.src_: s_states,
                       self.model.src_len_: s_len,
                       self.model.action_idx_: at_id_in})
        t2_end = ctime()

        expected_q = np.zeros_like(reward)
        for i in range(len(expected_q)):
            expected_q[i] = reward[i]
            if not is_terminal[i]:
                s_argmax_q, _, valid_len = get_best_2D_q(
                    s_q_actions_dqn[i, :, :], self.hp.eos_id)
                expected_q[i] += gamma * np.mean(
                    s_q_actions_target[i, range(valid_len),
                                       s_argmax_q[:valid_len]])

        t3 = ctime()
        _, summaries, loss_eval, abs_loss = sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={self.model.src_: p_states,
                       self.model.src_len_: p_len,
                       self.model.b_weight_: b_weight,
                       self.model.action_idx_: at_id_in,
                       self.model.action_idx_out_: at_id,
                       self.model.action_len_: action_len,
                       self.model.expected_q_: expected_q})
        t3_end = ctime()

        self.memo.batch_update(b_idx, abs_loss)

        if t % 1000 == 0:
            self.debug('t1: {}, t2: {}, t3: {}'.format(
                t1_end - t1, t2_end - t2, t3_end - t3))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)
