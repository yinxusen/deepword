import random
import sys
import time
import traceback
from os.path import join as pjoin
from queue import Queue
from threading import Thread
from typing import Tuple, List, Union, Any

import numpy as np
import tensorflow as tf
from tensorflow import Session
from tensorflow.contrib.training import HParams
from tensorflow.summary import FileWriter
from tensorflow.train import Saver
from tqdm import trange

from deeptextworld.action import ActionCollector
from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.agents.utils import ActionMaster
from deeptextworld.agents.utils import Memolet, pad_action
from deeptextworld.agents.utils import batch_dqn_input, batch_drrn_action_input
from deeptextworld.agents.utils import bert_commonsense_input
from deeptextworld.agents.utils import get_best_batch_ids
from deeptextworld.agents.utils import sample_batch_ids
from deeptextworld.hparams import save_hparams
from deeptextworld.students.utils import get_action_idx_pair
from deeptextworld.trajectory import Trajectory
from deeptextworld.utils import eprint, flatten
from deeptextworld.utils import model_name2clazz


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def set(self, key, val):
        self.__dict__[key] = val

    def get(self, key):
        return self.__dict__[key]


class StudentLearner(object):
    def __init__(
            self, hp: HParams, model_dir: str, train_data_dir: str) -> None:
        # prefix should match BaseAgent
        self.tjs_prefix = "trajectories"
        self.action_prefix = "actions"
        self.memo_prefix = "memo"

        self.model_dir = model_dir
        self.train_data_dir = train_data_dir
        self.load_from = pjoin(self.model_dir, "last_weights")
        self.ckpt_prefix = pjoin(self.load_from, "after-epoch")
        self.hp, self.tokenizer = BaseAgent.init_tokens(hp)
        save_hparams(hp, pjoin(model_dir, "hparams.json"))
        (self.sess, self.model, self.saver, self.sw, self.train_steps,
         self.queue) = self.prepare_training()

    def get_compatible_snapshot_tag(self) -> List[int]:

        action_tags = BaseAgent.get_path_tags(
            self.train_data_dir, self.action_prefix)
        memo_tags = BaseAgent.get_path_tags(
            self.train_data_dir, self.memo_prefix)
        tjs_tags = BaseAgent.get_path_tags(
            self.train_data_dir, self.tjs_prefix)

        valid_tags = set(action_tags)
        valid_tags.intersection_update(memo_tags)
        valid_tags.intersection_update(tjs_tags)

        return list(valid_tags)

    def get_combined_data_path(self) -> List[Tuple[str, str, str]]:
        valid_tags = self.get_compatible_snapshot_tag()
        combined_data_path = []
        for tag in sorted(valid_tags, key=lambda k: random.random()):
            combined_data_path.append(
                (pjoin(self.train_data_dir,
                       "{}-{}.npz".format(self.tjs_prefix, tag)),
                 pjoin(self.train_data_dir,
                       "{}-{}.npz".format(self.action_prefix, tag)),
                 pjoin(self.train_data_dir,
                       "{}-{}.npz".format(self.memo_prefix, tag))))
        return combined_data_path

    def prepare_model(
            self, device_placement: str
    ) -> Tuple[Session, Any, Saver, int]:
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

    def load_snapshot(
            self, memo_path: str, tjs_path: str, action_path: str
    ) -> Tuple[List[Tuple], Trajectory[ActionMaster], ActionCollector]:
        memory = np.load(memo_path, allow_pickle=True)["data"]
        memory = list(filter(lambda x: isinstance(x, Memolet), memory))

        tjs = Trajectory[ActionMaster](self.hp.num_turns)
        tjs.load_tjs(tjs_path)

        actions = ActionCollector(
            tokenizer=self.tokenizer,
            n_tokens=self.hp.n_tokens_per_action,
            unk_val_id=self.hp.unk_val_id,
            padding_val_id=self.hp.padding_val_id,
            pad_eos=False,
            eos_id=self.hp.eos_id)
        actions.load_actions(action_path)
        return memory, tjs, actions

    def add_batch(
            self, combined_data_path: List[Tuple[str, str, str]],
            queue: Queue) -> None:
        while True:
            for tp, ap, mp, in sorted(
                    combined_data_path, key=lambda k: random.random()):
                memory, tjs, action_collector = self.load_snapshot(mp, tp, ap)
                random.shuffle(memory)
                i = 0
                while i < len(memory) // self.hp.batch_size:
                    ss = i * self.hp.batch_size
                    ee = min((i + 1) * self.hp.batch_size, len(memory))
                    batch_memory = memory[ss:ee]
                    try:
                        queue.put(self.prepare_data(
                            batch_memory, tjs, action_collector),
                        )
                    except Exception as e:
                        eprint("add_batch error: {}".format(e))
                        traceback.print_tb(e.__traceback__)
                        raise RuntimeError()
                    i += 1

    def prepare_data(
            self,
            b_memory: List[Union[Tuple, Memolet]],
            tjs: Trajectory[ActionMaster],
            action_collector: ActionCollector) -> Tuple:
        """
        Given a batch of memory, tjs, and action collector, create a tuple
        of data for training.

        :param b_memory:
        :param tjs:
        :param action_collector:
        :return: Tuple of data, the train_impl knows the details
        """
        raise NotImplementedError()

    def prepare_training(
            self
    ) -> Tuple[Session, Any, Saver, FileWriter, int, Queue]:
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

    def train(self, n_epochs: int) -> None:
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
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(
                        exc_type, exc_value, exc_traceback, limit=None,
                        file=sys.stdout)
                    break
            self.saver.save(
                self.sess, self.ckpt_prefix,
                global_step=tf.train.get_or_create_global_step(
                    graph=self.model.graph))
            eprint("finish and save {} epoch".format(et))
            if not data_in_queue:
                break
        return

    def train_impl(self, data: Tuple, train_step: int) -> None:
        """
        Train the model one time given data.

        :param data:
        :param train_step:
        :return:
        """
        raise NotImplementedError()


class DRRNLearner(StudentLearner):
    def __init__(
            self, hp: HParams, model_dir: str, train_data_dir: str) -> None:
        super(DRRNLearner, self).__init__(hp, model_dir, train_data_dir)

    def train_impl(self, data, train_step):
        (p_states, p_len, actions, action_len, actions_repeats, expected_qs
         ) = data
        _, summaries = self.sess.run(
            [self.model.train_op, self.model.train_summary_op],
            feed_dict={
                self.model.src_: p_states,
                self.model.src_len_: p_len,
                self.model.actions_: actions,
                self.model.actions_len_: action_len,
                self.model.actions_repeats_: actions_repeats,
                self.model.action_idx_: np.arange(len(expected_qs)),
                self.model.expected_q_: expected_qs,
                self.model.b_weight_: [1.]})
        self.sw.add_summary(summaries, train_step)

    def prepare_data(self, b_memory, tjs, action_collector):
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        expected_qs = flatten([list(m.q_actions) for m in b_memory])

        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, _ = batch_dqn_input(
            states, self.tokenizer, self.hp.num_tokens, self.hp.padding_val_id)
        action_len = (
            [action_collector.get_action_len(gid) for gid in game_id])
        action_matrix = (
            [action_collector.get_action_matrix(gid) for gid in game_id])
        actions, action_len, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)
        return (
            p_states, p_len, actions, action_len, actions_repeats, expected_qs)


class GenLearner(StudentLearner):
    def train_impl(self, data, train_step):
        (p_states, p_len, master_mask, actions_in, actions_out, action_len
         ) = data
        _, summaries = self.sess.run(
            [self.model.train_seq2seq_op, self.model.train_seq2seq_summary_op],
            feed_dict={
                self.model.src_: p_states,
                self.model.src_len_: p_len,
                self.model.src_seg_: master_mask,
                self.model.action_idx_: actions_in,
                self.model.action_idx_out_: actions_out,
                self.model.action_len_: action_len,
                self.model.b_weight_: [1.]})
        self.sw.add_summary(summaries, train_step)

    def prepare_data(self, b_memory, tjs, action_collector):
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        expected_qs = np.asarray(flatten([list(m.q_actions) for m in b_memory]))
        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, master_mask = batch_dqn_input(
            states, self.tokenizer, self.hp.num_tokens, self.hp.padding_val_id)
        action_len = (
            [action_collector.get_action_len(gid) for gid in game_id])
        action_matrix = (
            [action_collector.get_action_matrix(gid) for gid in game_id])
        actions, action_len, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)
        best_q_idx = get_best_batch_ids(expected_qs, actions_repeats)
        actions_in, actions_out, action_len = get_action_idx_pair(
            actions[best_q_idx], action_len[best_q_idx],
            self.hp.sos_id, self.hp.eos_id)
        return p_states, p_len, master_mask, actions_in, actions_out, action_len


class GenConcatActionsLearner(GenLearner):
    def prepare_data(self, b_memory, tjs, action_collector):
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        admissible_action_mask = [m.sys_action_mask for m in b_memory]

        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, master_mask = batch_dqn_input(
            states, self.tokenizer, self.hp.num_tokens, self.hp.padding_val_id)

        actions = [
            " ; ".join(sorted(list(
                np.asarray(action_collector.get_actions(gid))[mid])))
            for gid, mid in zip(game_id, admissible_action_mask)]
        action_idx = [
            self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(concat_actions))
            for concat_actions in actions]
        action_len = np.asarray([len(x) for x in action_idx])
        max_concat_action_len = np.max(action_len)
        action_matrix = np.asarray(
            [pad_action(x, max_concat_action_len, self.hp.padding_val_id)
             for x in action_idx])

        actions_in, actions_out, action_len = get_action_idx_pair(
            action_matrix, action_len, self.hp.sos_id, self.hp.eos_id)
        return (
            p_states, p_len, master_mask, actions_in, actions_out, action_len)


class BertLearner(StudentLearner):
    def train_impl(self, data, train_step):
        inp, seg_tj_action, inp_len, swag_labels = data
        eprint(inp)
        eprint(seg_tj_action)
        eprint(inp_len)
        eprint(swag_labels)
        _, summaries = self.sess.run(
            [self.model.swag_train_op, self.model.swag_train_summary_op],
            feed_dict={
                self.model.src_: inp,
                self.model.src_len_: inp_len,
                self.model.seg_tj_action_: seg_tj_action,
                self.model.swag_labels_: swag_labels
                })
        self.sw.add_summary(summaries, train_step)

    def prepare_data(self, b_memory, tjs, action_collector):
        n_classes = 4
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        expected_qs = np.asarray(flatten([list(m.q_actions) for m in b_memory]))
        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, _ = batch_dqn_input(
            states, self.tokenizer, self.hp.num_tokens, self.hp.padding_val_id)
        action_len = (
            [action_collector.get_action_len(gid) for gid in game_id])
        action_matrix = (
            [action_collector.get_action_matrix(gid) for gid in game_id])
        actions, action_len, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)
        batch_q_idx = sample_batch_ids(
            expected_qs, actions_repeats, k=n_classes)

        # [CLS] + [trajectory] + [SEP] + [action] + [SEP]
        max_allowed_trajectory_size = (
            self.hp.num_tokens - 3 - self.hp.n_tokens_per_action)
        # fetch pre-trajectory
        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, _ = batch_dqn_input(
            states, self.tokenizer, max_allowed_trajectory_size,
            self.hp.padding_val_id)
        batch_size = len(p_states)

        action_len = action_len[batch_q_idx].reshape((batch_size, n_classes))
        actions = actions[batch_q_idx].reshape((batch_size, n_classes, -1))
        swag_labels = np.zeros((len(actions), ), dtype=np.int)

        processed_input = [
            bert_commonsense_input(
                am, al, tj, tj_len, self.hp.sep_val_id, self.hp.cls_val_id,
                self.hp.num_tokens)
            for am, al, tj, tj_len
            in zip(list(actions), list(action_len), p_states, p_len)]

        inp = np.concatenate([a[0] for a in processed_input], axis=0)
        seg_tj_action = np.concatenate([a[1] for a in processed_input], axis=0)
        inp_len = np.concatenate([a[2] for a in processed_input], axis=0)

        return inp, seg_tj_action, inp_len, swag_labels
