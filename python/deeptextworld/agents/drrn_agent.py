import numpy as np
from bert.tokenization import FullTokenizer
from textworld import EnvInfos

from deeptextworld import drrn_model
from deeptextworld.agents.base_agent import BaseAgent, IDX_GR2GA
from deeptextworld.dqn_func import get_random_1Daction, get_best_1Daction, \
    get_best_1D_q
from deeptextworld.hparams import copy_hparams
from deeptextworld.utils import ctime, load_vocab, get_token2idx


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

    def get_an_eps_action(self, action_mask, go_room_action_ids):
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
            action_matrix, actions_len = self.change_go_actions(
                go_room_action_ids,
                self.action_collector.get_action_matrix(-1),
                self.game_id)
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

    def create_eval_model_instance(self):
        model_creator = getattr(drrn_model, self.hp.model_creator)
        model = drrn_model.create_eval_model(model_creator, self.hp)
        return model

    def change_go_actions(
            self, go_room_actions_ids, go_room_action_matrix, game_id):
        """
        change `go west` to `go west to kitchen`, etc.
        make sure go east/ go west/ go north/ go south
          are indexed as 0, 1, 2, 3, respectively.
        deepcopy is required since we need to change the action matrix and lens
        """
        am = np.copy(self.action_collector.get_action_matrix(game_id))
        al = np.copy(self.action_collector.get_action_len(game_id))
        for aid in go_room_actions_ids:
            am[IDX_GR2GA[aid], :] = go_room_action_matrix[aid]
            al[IDX_GR2GA[aid]] = 4
        return am, al

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
        go_room_action_ids = [m[0].go_room_action_ids for m in b_memory]
        next_go_room_action_ids = [m[0].next_go_room_action_ids for m in b_memory]


        action_mask_t = self.fromBytes(action_mask)
        action_mask_t1 = self.fromBytes(next_action_mask)

        p_states, s_states, p_len, s_len =\
            self.tjs.fetch_batch_states_pair(trajectory_id, state_id)

        go_rooms = self.action_collector.get_action_matrix(eid=-1)

        action_matrix_len_t1 = (
            [self.change_go_actions(next_go_room_action_ids[i], go_rooms, gid)
             for i, gid in enumerate(game_id)])
        action_matrix_t1 = [aml[0] for aml in action_matrix_len_t1]
        actions_len_t1 = [aml[1] for aml in action_matrix_len_t1]

        t2 = ctime()
        s_q_actions_target = target_sess.run(
            self.target_model.q_actions,
            feed_dict={self.target_model.src_: s_states,
                       self.target_model.src_len_: s_len,
                       self.target_model.actions_: action_matrix_t1,
                       self.target_model.actions_len_: actions_len_t1,
                       self.target_model.actions_mask_: action_mask_t1})

        s_q_actions_dqn = sess.run(
            self.model.q_actions,
            feed_dict={self.model.src_: s_states,
                       self.model.src_len_: s_len,
                       self.model.actions_: action_matrix_t1,
                       self.model.actions_len_: actions_len_t1,
                       self.model.actions_mask_: action_mask_t1})
        t2_end = ctime()

        expected_q = np.zeros_like(reward)
        for i in range(len(expected_q)):
            expected_q[i] = reward[i]
            if not is_terminal[i]:
                s_argmax_q, _ = get_best_1D_q(
                    s_q_actions_dqn[i, :], mask=action_mask_t1[i])
                expected_q[i] += gamma * s_q_actions_target[i, s_argmax_q]

        action_matrix_len_t = (
            [self.change_go_actions(go_room_action_ids[i], go_rooms, gid)
             for i, gid in enumerate(game_id)])
        action_matrix_t = [aml[0] for aml in action_matrix_len_t]
        actions_len_t = [aml[1] for aml in action_matrix_len_t]

        for i in range(len(b_idx)):
            if action_id[i] < 4:
                self.info("action_idx: {}".format(action_id[i]))
                self.info("ids_map: {}".format(go_room_action_ids[i]))
                self.info("action_matrix_t: {}".format(action_matrix_t[i][:10]))
                self.info("ids_map_t1: {}".format(next_go_room_action_ids[i]))
                self.info("action_matrix_t1: {}".format(action_matrix_t1[i][:10]))

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
                       self.model.actions_: action_matrix_t,
                       self.model.actions_len_: actions_len_t})
        t3_end = ctime()

        self.memo.batch_update(b_idx, abs_loss)

        self.info('loss: {}'.format(loss_eval))
        self.debug('t1: {}, t2: {}, t3: {}'.format(
            t1_end-t1, t2_end-t2, t3_end-t3))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)


class BertDRRNAgent(DRRNAgent):
    """
    """
    def __init__(self, hp, model_dir):
        super(BertDRRNAgent, self).__init__(hp, model_dir)
        self.tokenizer = FullTokenizer(
            vocab_file=hp.vocab_file, do_lower_case=True)

    def init_tokens(self, hp):
        """
        :param hp:
        :return:
        """
        new_hp = copy_hparams(hp)
        # make sure that padding_val is indexed as 0.
        tokens = list(load_vocab(hp.vocab_file))
        print(tokens[:10])
        token2idx = get_token2idx(tokens)
        new_hp.set_hparam('vocab_size', len(tokens))
        new_hp.set_hparam('padding_val_id', token2idx[hp.padding_val])
        new_hp.set_hparam('unk_val_id', token2idx[hp.unk_val])
        # bert specific tokens
        new_hp.set_hparam('cls_val_id', token2idx[hp.cls_val])
        new_hp.set_hparam('sep_val_id', token2idx[hp.sep_val])
        new_hp.set_hparam('mask_val_id', token2idx[hp.mask_val])
        return new_hp, tokens, token2idx

    def tokenize(self, master):
        return ' '.join([t.lower() for t in self.tokenizer.tokenize(master)])
