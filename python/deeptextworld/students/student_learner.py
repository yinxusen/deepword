import random
import time
from os.path import join as pjoin
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow as tf
from tqdm import trange

from deeptextworld.action import ActionCollector
from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.hparams import save_hparams
from deeptextworld.models.dqn_func import get_batch_best_1D_idx
from deeptextworld.students.utils import get_action_idx_pair
from deeptextworld.students.utils import model_name2clazz
from deeptextworld.trajectory import RawTextTrajectory
from deeptextworld.utils import flatten, eprint


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def set(self, key, val):
        self.__dict__[key] = val


class StudentLearner(object):
    def __init__(self, hp, model_dir, train_data_dir, n_data):
        self.model_dir = model_dir
        self.train_data_dir = train_data_dir
        self.n_data = n_data
        self.load_from = pjoin(self.model_dir, "last_weights")
        self.ckpt_prefix = pjoin(self.load_from, "after-epoch")
        self.hp, self.tokenizer = BaseAgent.init_tokens(hp)
        save_hparams(
            hp, pjoin(model_dir, "hparams.json"), use_relative_path=True)
        (self.sess, self.model, self.saver, self.sw, self.train_steps,
         self.queue) = self.prepare_training()

    def get_combined_data_path(self):
        tjs_prefix = "raw-trajectories"
        action_prefix = "actions"
        memo_prefix = "memo"
        hs2tj_prefix = "hs2tj"

        combined_data_path = []
        for i in sorted(range(self.n_data), key=lambda k: random.random()):
            combined_data_path.append(
                (pjoin(self.train_data_dir,
                       "{}-{}.npz".format(tjs_prefix, i)),
                 pjoin(self.train_data_dir,
                       "{}-{}.npz".format(action_prefix, i)),
                 pjoin(self.train_data_dir,
                       "{}-{}.npz".format(memo_prefix, i)),
                 pjoin(self.train_data_dir,
                       "{}-{}.npz".format(hs2tj_prefix, i))))
        return combined_data_path

    def prepare_model(self, device_placement):
        model_clazz = model_name2clazz(self.hp.model_creator)
        model = model_clazz.get_train_student_model(
            hp=self.hp,
            device_placement=device_placement)
        conf = tf.ConfigProto(
            log_device_placement=False, allow_soft_placement=True)
        sess = tf.Session(graph=model.graph, config=conf)
        with model.graph.as_default():
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(
                max_to_keep=self.hp.max_snapshot_to_keep,
                save_relative_paths=True)
            global_step = tf.train.get_or_create_global_step()

        try:
            ckpt_path = tf.train.latest_checkpoint(self.load_from)
            saver.restore(sess, ckpt_path)
            trained_steps = sess.run(global_step)
            eprint("load student from ckpt: {}".format(ckpt_path))
        except Exception as e:
            eprint("load model failed: {}".format(e))
            trained_steps = 0
        return sess, model, saver, trained_steps

    def load_snapshot(self, memo_path, raw_tjs_path, action_path):
        memory = np.load(memo_path)['data']
        memory = list(filter(lambda x: isinstance(x, tuple), memory))

        tjs = RawTextTrajectory(self.hp)
        tjs.load_tjs(raw_tjs_path)

        actions = ActionCollector(
            self.tokenizer, self.hp.n_actions, self.hp.n_tokens_per_action,
            unk_val_id=self.hp.unk_val_id,
            padding_val_id=self.hp.padding_val_id)
        actions.load_actions(action_path)
        return memory, tjs, actions

    def add_batch(self, combined_data_path, queue):
        while True:
            for tp, ap, mp, hs in sorted(
                    combined_data_path, key=lambda k: random.random()):
                memory, tjs, action_collector = self.load_snapshot(mp, tp, ap)
                random.shuffle(memory)
                i = 0
                while i < len(memory) // self.hp.batch_size:
                    ss = i * self.hp.batch_size
                    ee = min((i + 1) * self.hp.batch_size, len(memory))
                    batch_memory = memory[ss:ee]
                    queue.put(self.prepare_data(
                        batch_memory, tjs, action_collector),
                    )
                    i += 1

    def prepare_data(self, b_memory, tjs, action_collector):
        """
        Given a batch of memory, tjs, and action collector, create a tuple
        of data for training.

        :param b_memory:
        :param tjs:
        :param action_collector:
        :return:
        """
        raise NotImplementedError()

    def prepare_training(self):
        sess, model, saver, train_steps = self.prepare_model("/device:GPU:0")

        # save the very first model to verify weight has been loaded
        if train_steps == 0:
            saver.save(
                sess, self.ckpt_prefix,
                global_step=tf.train.get_or_create_global_step(
                    graph=model.graph))
        else:
            pass

        sw_path = pjoin(self.model_dir, "summaries", "train")
        sw = tf.summary.FileWriter(sw_path, sess.graph)

        queue = Queue(maxsize=100)
        t = Thread(
            target=self.add_batch,
            args=(self.get_combined_data_path(), queue))
        t.setDaemon(True)
        t.start()

        return sess, model, saver, sw, train_steps, queue

    def train(self, n_epochs):
        wait_times = 10
        while wait_times > 0 and self.queue.empty():
            eprint("waiting data ... (retry times: {})".format(wait_times))
            time.sleep(10)
            wait_times -= 1

        epoch_size = self.hp.save_gap_t

        eprint("start training")
        data_in_queue = True
        for et in trange(n_epochs, ascii=True, desc="epoch"):
            for it in trange(epoch_size, ascii=True, desc="step"):
                try:
                    data = self.queue.get(timeout=100)
                    self.train_impl(
                        data, self.train_steps + et * epoch_size + it)
                except Exception as e:
                    data_in_queue = False
                    eprint("no more data: {}".format(e))
                    break
            self.saver.save(
                self.sess, self.ckpt_prefix,
                global_step=tf.train.get_or_create_global_step(
                    graph=self.model.graph))
            eprint("finish and save {} epoch".format(et))
            if not data_in_queue:
                break
        return

    def train_impl(self, data, train_step):
        """
        Train the model one time given data.

        :param data:
        :param train_step:
        :return:
        """
        raise NotImplementedError()

    def str2idx(self, sentence):
        """
        Convert sentence into indices.
        :param sentence:
        :return:
        """
        return self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(sentence))

    def prepare_trajectory(self, trajectory):
        """
        Convert a trajectory into indices to a specific length of hp.num_tokens.

        trim the result indices from head if it contains more than that tokens,
        or pad with padding_val_id if less than.

        :param trajectory:
          list of master-player pairs
        :return:
        """
        indices = flatten([self.str2idx(s) for s in trajectory])
        if len(indices) > self.hp.num_tokens:
            indices = indices[len(indices) - self.hp.num_tokens:]
        else:
            indices = indices + [self.hp.padding_val_id] * (
                    self.hp.num_tokens - len(indices))
        return indices


class DRRNLearner(StudentLearner):
    def train_impl(self, data, train_step):
        (p_states, p_len, action_matrix, action_mask_t, action_len,
         expected_qs) = data
        _, summaries = self.sess.run(
            [self.model.train_op, self.model.train_summary_op],
            feed_dict={
                self.model.src_: p_states,
                self.model.src_len_: p_len,
                self.model.actions_mask_: action_mask_t,
                self.model.actions_: action_matrix,
                self.model.actions_len_: action_len,
                self.model.expected_qs_: expected_qs})
        self.sw.add_summary(summaries, train_step)
        return

    def prepare_data(self, b_memory, tjs, action_collector):
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        expected_qs = [m.q_actions for m in b_memory]
        action_mask_t = BaseAgent.from_bytes(action_mask)

        states = tjs.fetch_batch_states(trajectory_id, state_id)
        p_states = [self.prepare_trajectory(s) for s in states]
        p_len = [len(state) for state in p_states]
        action_len = (
            [action_collector.get_action_len(gid) for gid in game_id])
        max_action_len = np.max(action_len)
        action_matrix = (
            [action_collector.get_action_matrix(gid)[:, :max_action_len]
             for gid in game_id])

        return (
            p_states, p_len, action_matrix, action_mask_t, action_len,
            expected_qs)


class GenLearner(StudentLearner):
    def train_impl(self, data, train_step):
        (p_states, p_len, actions_in, actions_out, action_len,
         expected_qs, b_weights) = data
        _, summaries = self.sess.run(
            [self.model.train_seq2seq_op, self.model.train_seq2seq_summary_op],
            feed_dict={self.model.src_: p_states,
                       self.model.src_len_: p_len,
                       self.model.action_idx_: actions_in,
                       self.model.action_idx_out_: actions_out,
                       self.model.action_len_: action_len,
                       self.model.b_weight_: b_weights})
        self.sw.add_summary(summaries, train_step)
        return

    def prepare_data(self, b_memory, tjs, action_collector):
        """
        ("tid", "sid", "gid", "aid", "reward", "is_terminal",
         "action_mask", "next_action_mask", "q_actions")
        """
        trajectory_id = [m[0] for m in b_memory]
        state_id = [m[1] for m in b_memory]
        game_id = [m[2] for m in b_memory]
        action_mask = [m[6] for m in b_memory]
        expected_qs = [m[8] for m in b_memory]
        best_q_idx = get_batch_best_1D_idx(
            expected_qs, BaseAgent.from_bytes(action_mask))

        states = tjs.fetch_batch_states(trajectory_id, state_id)
        p_states = [self.prepare_trajectory(s) for s in states]
        p_len = [len(state) for state in p_states]

        action_len = np.asarray(
            [action_collector.get_action_len(gid)[mid]
             for gid, mid in zip(game_id, best_q_idx)])
        actions = np.asarray(
            [action_collector.get_action_matrix(gid)[mid, :]
             for gid, mid in zip(game_id, best_q_idx)])
        actions_in, actions_out, action_len = get_action_idx_pair(
            actions, action_len, self.tokenizer.vocab["<S>"],
            self.tokenizer.vocab["</S>"])
        b_weights = np.ones_like(action_len, dtype="float32")
        return (
            p_states, p_len,
            actions_in, actions_out, action_len, expected_qs, b_weights)


class GenPreTrainLearner(GenLearner):
    def prepare_data(self, b_memory, tjs, action_collector):
        """
            ("tid", "sid", "gid", "aid", "reward", "is_terminal",
             "action_mask", "next_action_mask", "q_actions")
            """
        trajectory_id = [m[0] for m in b_memory]
        state_id = [m[1] for m in b_memory]
        game_id = [m[2] for m in b_memory]
        action_mask = [m[6] for m in b_memory]
        expected_qs = [m[8] for m in b_memory]
        action_mask_t = list(BaseAgent.from_bytes(action_mask))
        # mask_idx = list(map(lambda m: np.where(m == 1)[0], action_mask_t))
        selected_mask_idx = list(map(
            lambda m: np.random.choice(np.where(m == 1)[0], size=[2, ]),
            action_mask_t))

        states = tjs.fetch_batch_states(trajectory_id, state_id)
        p_states = [self.prepare_trajectory(s) for s in states]
        p_len = [len(state) for state in p_states]

        action_len = np.concatenate(
            [action_collector.get_action_len(gid)[mid]
             for gid, mid in zip(game_id, selected_mask_idx)], axis=0)
        actions = np.concatenate(
            [action_collector.get_action_matrix(gid)[mid, :]
             for gid, mid in zip(game_id, selected_mask_idx)], axis=0)
        actions_in, actions_out, action_len = get_action_idx_pair(
            actions, action_len, self.tokenizer.vocab["<S>"],
            self.tokenizer.vocab["</S>"])
        # repeats = np.sum(action_mask_t, axis=1)
        repeats = 2
        repeated_p_states = np.repeat(p_states, repeats, axis=0)
        repeated_p_len = np.repeat(p_len, repeats, axis=0)
        expected_qs = np.concatenate(
            [qs[mid] for qs, mid in zip(expected_qs, selected_mask_idx)],
            axis=0)
        b_weights = np.ones_like(action_len, dtype="float32")
        return (
            repeated_p_states, repeated_p_len,
            actions_in, actions_out, action_len, expected_qs, b_weights)
