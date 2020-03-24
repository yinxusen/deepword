import os
import random
import traceback
from csv import reader
from os.path import join as pjoin
from queue import Queue
from threading import Thread
from typing import Tuple, List, Any

import fire
import numpy as np
import tensorflow as tf
from tensorflow import Session
from tensorflow.summary import FileWriter
from tensorflow.train import Saver

from deeptextworld.agents.utils import Tokenizer
from deeptextworld.agents.utils import bert_commonsense_input
from deeptextworld.agents.utils import pad_str_ids
from deeptextworld.students.student_learner import BertLearner, CMD
from deeptextworld.students.train_eval_framework import TrainEval
from deeptextworld.utils import eprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

MAX_ENDING_STR_LEN = 100


def process_one_line(
        tokenizer: Tokenizer,
        start_str: str,
        ending_str: List[str],
        max_start_len: int) -> Tuple[List[int], int, np.ndarray, np.ndarray]:
    tj_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(start_str))
    tj_len = len(tj_ids)
    padded_tj_ids = pad_str_ids(
        tj_ids, max_size=max_start_len, padding_val_id=0)
    actions = [
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(a))
        for a in ending_str]
    action_len = np.asarray([len(a) for a in actions], dtype=np.int32)
    max_action_size = min(np.max(action_len), MAX_ENDING_STR_LEN)
    action_matrix = np.asarray(
        [pad_str_ids(a, max_size=max_action_size, padding_val_id=0)
         for a in actions], dtype=np.int32)
    return padded_tj_ids, tj_len, action_matrix, action_len


def load_swag_data(
        fn_swag: str) -> Tuple[List[str], List[np.ndarray], List[int]]:
    """read swag data, make sure every start sentence has four ends"""
    with open(fn_swag, "r") as f:
        lines = list(reader(f.readlines()))[1:]  # remove the first line

    lines = np.asarray(lines, dtype=np.object)
    labels = list(lines[:, -1].astype(np.int32))
    start_str = list(lines[:, 3])
    ending_str = list(lines[:, 7:11])

    return start_str, ending_str, labels


def get_bert_input(
        start_str: List[str], ending_str: List[np.ndarray],
        tokenizer: Tokenizer, sep_val_id, cls_val_id, num_tokens
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inp = []
    seg_tj_action = []
    inp_size = []
    num_tokens_tj = num_tokens - MAX_ENDING_STR_LEN - 3

    for tj, actions in zip(start_str, ending_str):
        tj_ids, tj_len, action_matrix, action_len = process_one_line(
            tokenizer, tj, list(actions), num_tokens_tj)
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


class SwagLearner(BertLearner):
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
            args=("{}/data/train.csv".format(self.train_data_dir), queue))
        t.setDaemon(True)
        t.start()

        return sess, model, saver, sw, train_steps, queue

    def add_batch(self, swag_path: str, queue: Queue) -> None:
        start_str, ending_str, labels = load_swag_data(swag_path)
        data = list(zip(start_str, ending_str, labels))
        while True:
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
                        self.hp.num_tokens)
                    queue.put((inp, seg_tj_action, inp_size, batch_labels))
                except Exception as e:
                    eprint("add_batch error: {}".format(e))
                    traceback.print_tb(e.__traceback__)
                    raise RuntimeError()
                i += 1


if __name__ == "__main__":
    cmd_args = CMD(
        model_dir="",
        model_creator="BertCommonsenseModel",
        num_tokens=500,
        batch_size=8,
        save_gap_t=5000,
        learning_rate=5e-5,
        tokenizer_type="BERT",
        max_snapshot_to_keep=100
    )
    train_eval = TrainEval(cmd_args, SwagLearner)
    fire.Fire(train_eval)
