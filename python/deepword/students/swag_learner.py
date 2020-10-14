import os
import random
import time
import traceback
from csv import reader
from os import path
from queue import Queue
from threading import Thread
from typing import Tuple, List, Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow import Session
from tensorflow.summary import FileWriter
from tensorflow.train import Saver

from deepword.agents.utils import bert_commonsense_input
from deepword.students.student_learner import NLUClassificationLearner
from deepword.students.utils import align_batch_str
from deepword.tokenizers import Tokenizer
from deepword.utils import eprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


def process_one_line(
        tokenizer: Tokenizer,
        start_str: str,
        ending_str: List[str],
        max_start_len: int,
        max_end_len: int) -> Tuple[List[int], int, np.ndarray, np.ndarray]:
    tj_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(start_str))
    tj_len = len(tj_ids)
    aligned_tj = align_batch_str(
        [tj_ids], str_len_allowance=max_start_len, padding_val_id=0,
        valid_len=[tj_len])
    tj_ids = aligned_tj[0][0]
    tj_len = aligned_tj[1][0]

    actions = [
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(a))
        for a in ending_str]
    action_matrix, action_len = align_batch_str(
        actions, str_len_allowance=max_end_len, padding_val_id=0,
        valid_len=[len(a) for a in actions])

    return tj_ids, tj_len, action_matrix, action_len


def load_swag_data(
        fn_swag: str) -> Tuple[List[str], List[np.ndarray], List[int]]:
    """read swag data, make sure every start sentence has four ends"""
    with open(fn_swag, "r") as f:
        lines = list(reader(f.readlines()))[1:]  # remove the first line

    assert len(lines[0]) == 12, "use train.csv or val.csv which contain labels"

    lines = np.asarray(lines, dtype=np.object)
    labels = list(lines[:, -1].astype(np.int32))
    start_str = list(lines[:, 3])
    ending_str = list(lines[:, 7:11])

    return start_str, ending_str, labels


def get_bert_input(
        start_str: List[str], ending_str: List[np.ndarray],
        tokenizer: Tokenizer, sep_val_id, cls_val_id,
        num_tokens, n_tokens_per_action
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO: here we use n_tokens_per_actions for end str size
    """
    inp = []
    seg_tj_action = []
    inp_size = []
    num_tokens_tj = num_tokens - n_tokens_per_action - 3

    for tj, actions in zip(start_str, ending_str):
        tj_ids, tj_len, action_matrix, action_len = process_one_line(
            tokenizer, tj, list(actions), num_tokens_tj, n_tokens_per_action)
        _inp, _seg_tj_action, _inp_size = bert_commonsense_input(
            action_matrix, action_len, tj_ids, tj_len,
            sep_val_id, cls_val_id, num_tokens)
        inp.append(_inp)
        seg_tj_action.append(_seg_tj_action)
        inp_size.append(_inp_size)

    inp = np.concatenate(inp, axis=0)
    seg_tj_action = np.concatenate(seg_tj_action, axis=0)
    inp_size = np.concatenate(inp_size, axis=0)
    return inp, seg_tj_action, inp_size


class SwagLearner(NLUClassificationLearner):
    def _prepare_training(
            self
    ) -> Tuple[Session, Any, Saver, FileWriter, int, Queue]:
        sess, model, saver, train_steps = self._prepare_model("/device:GPU:0")

        # save the very first model to verify weight has been loaded
        if train_steps == 0:
            saver.save(
                sess, self.ckpt_prefix,
                global_step=tf.train.get_or_create_global_step(
                    graph=model.graph))
        else:
            pass

        sw_path = path.join(self.model_dir, "summaries", "train")
        sw = tf.summary.FileWriter(sw_path, sess.graph)

        queue = Queue(maxsize=100)
        t = Thread(
            target=self._add_batch,
            args=("{}/data/train.csv".format(self.train_data_dir), queue))
        t.setDaemon(True)
        t.start()

        return sess, model, saver, sw, train_steps, queue

    def _prepare_test(
            self, device_placement: str = "/device:GPU:0",
            restore_from: Optional[str] = None
    ) -> Tuple[Session, Any, Saver, int, Queue]:
        sess, model, saver, train_steps = self._prepare_eval_model(
            device_placement, restore_from)
        queue = Queue(maxsize=100)
        t = Thread(
            target=self._add_batch,
            args=(self.eval_data_path, queue, False))
        t.setDaemon(True)
        t.start()
        return sess, model, saver, train_steps, queue

    def test(
            self, device_placement: str = "/device:GPU:0",
            restore_from: Optional[str] = None) -> Tuple[int, int]:
        if self.sess is None:
            (self.sess, self.model, self.saver, self.train_steps, self.queue
             ) = self._prepare_test(device_placement, restore_from)

        wait_times = 10
        while wait_times > 0 and self.queue.empty():
            eprint("waiting data ... (retry times: {})".format(wait_times))
            time.sleep(10)
            wait_times -= 1

        acc = 0
        total = 0
        eprint("start test")
        i = 0
        for data in iter(self.queue.get, None):
            inp, seg_tj_action, inp_len, _, swag_labels = data

            if i % 100 == 0:
                print("process a batch of {} .. {}".format(len(inp), i))
                print("partial acc.: {}".format(
                    acc * 1. / total if total else "Nan"))

            q_actions_t = self.sess.run(
                self.model.q_actions,
                feed_dict={
                    self.model.src_: inp,
                    self.model.src_len_: inp_len,
                    self.model.seg_tj_action_: seg_tj_action
                })
            q_actions_t = q_actions_t.reshape((len(swag_labels), -1))
            predicted_swag_labels = np.argmax(q_actions_t, axis=-1)
            acc += np.count_nonzero(predicted_swag_labels == swag_labels)
            total += len(swag_labels)
            i += 1
        return acc, total

    def _add_batch(
            self, swag_path: str,
            queue: Queue, training: bool = True,
            append_new_data: bool = True) -> None:
        start_str, ending_str, labels = load_swag_data(swag_path)
        data = list(zip(start_str, ending_str, labels))
        while True:
            if training:
                random.shuffle(data)
            i = 0
            while i < len(data) // self.hp.batch_size:
                ss = i * self.hp.batch_size
                ee = min((i + 1) * self.hp.batch_size, len(data))
                try:
                    batch_data = data[ss:ee]
                    batch_start_str = [x[0] for x in batch_data]
                    batch_ending_str = [x[1] for x in batch_data]
                    batch_labels = [x[2] for x in batch_data]

                    inp, seg_tj_action, inp_size = get_bert_input(
                        batch_start_str, batch_ending_str, self.tokenizer,
                        self.hp.sep_val_id, self.hp.cls_val_id,
                        self.hp.num_tokens, self.hp.n_tokens_per_action)
                    queue.put(
                        (inp, seg_tj_action, inp_size, None, batch_labels))
                except Exception as e:
                    eprint("add_batch error: {}".format(e))
                    traceback.print_tb(e.__traceback__)
                    raise RuntimeError()
                i += 1

            if not training:
                self.queue.put(None)
                break
