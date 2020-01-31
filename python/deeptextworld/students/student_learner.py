import os
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
from deeptextworld.trajectory import RawTextTrajectory
from deeptextworld.students.utils import names2clazz
from deeptextworld.utils import flatten, eprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class StudentLearner(object):
    def __init__(self, hp, model_dir, train_data_dir, n_data):
        self.model_dir = model_dir
        self.train_data_dir = train_data_dir
        self.n_data = n_data
        self.load_from = pjoin(self.model_dir, "last_weights")
        self.ckpt_prefix = pjoin(self.load_from, "after-epoch")
        self.hp, self.tokenizer = BaseAgent.init_tokens(hp)
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
        model_clazz = names2clazz(self.hp.model_creator)
        model = model_clazz.get_train_model(
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
                    data = self.queue.get(timeout=10)
                    self.train_impl(data)
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

    def train_impl(self, data):
        """
        Train the model one time given data.

        :param data:
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
