import math
import random
from collections import namedtuple
from os import path
from queue import Queue
from typing import Tuple, List, Any, Optional, Dict, Union, Generator

import numpy as np
import tensorflow as tf
from tensorflow import Session
from tensorflow.contrib.training import HParams
from tensorflow.summary import FileWriter
from tensorflow.train import Saver
from tensorflow.train import Saver
from tqdm import trange, tqdm

from deepword.action import ActionCollector
from deepword.agents.utils import Memolet
from deepword.agents.utils import get_path_tags
from deepword.students.student_learner import StudentLearner
from deepword.students.utils import ActionMasterStr
from deepword.trajectory import Trajectory


class SNNData(namedtuple(
    "SNNData", (
            "target_mids", "target_aids",
            "same_mids", "same_aids",
            "diff_mids", "diff_aids"))):
    pass


class SentenceLearner(StudentLearner):
    def _prepare_data(
            self, b_memory: List[Union[Tuple, Memolet]],
            tjs: Trajectory[ActionMasterStr],
            action_collector: ActionCollector) -> Tuple:
        raise NotImplementedError()

    def __init__(
            self, hp: HParams, model_dir: str, train_data_dir: Optional[str],
            eval_data_path: Optional[str] = None) -> None:
        super(SentenceLearner, self).__init__(
            hp, model_dir, train_data_dir, eval_data_path)

    def preprocess_input(self):
        valid_tags = self._get_compatible_snapshot_tag()
        for tag in valid_tags:
            tp = path.join(
                self.train_data_dir, "{}-{}.npz".format(self.tjs_prefix, tag))
            hsp = path.join(
                self.train_data_dir, "{}-{}.npz".format(self.hs2tj_prefix, tag))
            ap = path.join(
                self.train_data_dir,
                "{}-{}.npz".format(self.action_prefix, tag))
            mp = path.join(
                self.train_data_dir, "{}-{}.npz".format(self.memo_prefix, tag))

            memory, tjs, action_collector, hash_states2tjs = \
                self._load_snapshot(mp, tp, ap, hsp)

            # for every loaded snapshot, we sample SNN pairs
            # according to len(memory) / batch_size
            total_size = int(
                math.ceil(len(memory) * 1. / self.hp.batch_size))
            target_set, same_set, diff_set = self.get_snn_pairs(
                hash_states2tjs, tjs, total_size)
            target_aids, target_mids, target_tjs = self.get_snn_tjs(
                tjs, target_set)
            same_aids, same_mids, same_tjs = self.get_snn_tjs(tjs, same_set)
            diff_aids, diff_mids, diff_tjs = self.get_snn_tjs(tjs, diff_set)

            for i in range(min(len(target_tjs), 10)):
                self.info("target: {}, same: {}, diff: {}".format(
                    target_tjs[i], same_tjs[i], diff_tjs[i]))

            np.savez(
                "{}/snn-data-{}.npz".format(self.model_dir, tag),
                target_aids=target_aids,
                target_mids=target_mids,
                same_aids=same_aids,
                same_mids=same_mids,
                diff_aids=diff_aids,
                diff_mids=diff_mids)

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
        queue = Queue()  # empty queue
        return sess, model, saver, sw, train_steps, queue

    def snn_data_loader(
            self, data_path: str, batch_size: int, training: bool
    ) -> Generator[SNNData, None, None]:
        data_tags = sorted(get_path_tags(data_path, prefix="snn-data"))
        self.info("load snn tags: {}".format(data_tags))
        while True:
            for tag in tqdm(sorted(data_tags, key=lambda k: random.random())):
                self.info("load data from {}".format(tag))
                data = np.load(
                    path.join(data_path, "snn-data-{}.npz".format(tag)))
                target_aids = data["target_aids"]
                target_mids = data["target_mids"]
                same_aids = data["same_aids"]
                same_mids = data["same_mids"]
                diff_aids = data["diff_aids"]
                diff_mids = data["diff_mids"]
                for i in trange(len(target_aids) // batch_size):
                    ss = i * batch_size
                    ee = (i + 1) * batch_size
                    yield SNNData(
                        target_mids[ss: ee], target_aids[ss: ee],
                        same_mids[ss: ee], same_aids[ss: ee],
                        diff_mids[ss: ee], diff_aids[ss: ee])
            # only load data one time for evaluation
            if not training:
                self.info("snn data loader finished")
                break

    def train(self, n_epochs: int) -> None:
        if self.sess is None:
            (self.sess, self.model, self.saver, self.sw, self.train_steps,
             self.queue) = self._prepare_training()

        epoch_size = self.hp.save_gap_t
        data_loader = self.snn_data_loader(
            data_path=self.train_data_dir, batch_size=self.hp.batch_size,
            training=True)

        self.info("start training")
        data_in_queue = True
        for et in trange(n_epochs, ascii=True, desc="epoch"):
            for it in trange(epoch_size, ascii=True, desc="step"):
                self._train_impl(
                    next(data_loader), self.train_steps + et * epoch_size + it)
            self.saver.save(
                self.sess, self.ckpt_prefix,
                global_step=tf.train.get_or_create_global_step(
                    graph=self.model.graph))
            self.info("finish and save {} epoch".format(et))
            if not data_in_queue:
                break
        return

    def _train_impl(self, data: SNNData, train_step: int) -> None:
        _, summaries, loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss],
            feed_dict={
                self.model.target_master_: data.target_mids,
                self.model.same_master_: data.same_mids,
                self.model.diff_master_: data.diff_mids,
                self.model.target_action_: data.target_aids,
                self.model.same_action_: data.same_aids,
                self.model.diff_action_: data.diff_aids
            })
        self.sw.add_summary(summaries, train_step)

    def _prepare_test(
            self, device_placement: str = "/device:GPU:0",
            restore_from: Optional[str] = None
    ) -> Tuple[Session, Any, Saver, int, Queue]:
        sess, model, saver, train_steps = self._prepare_eval_model(
            device_placement, restore_from)
        queue = Queue()
        return sess, model, saver, train_steps, queue

    def test(
            self, device_placement: str = "/device:GPU:0",
            restore_from: Optional[str] = None) -> Tuple[int, int]:
        if self.sess is None:
            (self.sess, self.model, self.saver, self.train_steps, self.queue
             ) = self._prepare_test(device_placement, restore_from)

        data_loader = self.snn_data_loader(
            data_path=self.eval_data_path, batch_size=self.hp.batch_size,
            training=False)
        acc = 0
        total = 0
        self.info("start test")

        for data in data_loader:
            semantic_same = self.sess.run(
                self.model.semantic_same,
                feed_dict={
                    self.model.target_master_: data.target_mids,
                    self.model.same_master_: data.same_mids,
                    self.model.diff_master_: data.diff_mids,
                    self.model.target_action_: data.target_aids,
                    self.model.same_action_: data.same_aids,
                    self.model.diff_action_: data.diff_aids
                })

            acc += np.count_nonzero(
                semantic_same[: len(semantic_same) // 2] < 0)
            acc += np.count_nonzero(
                semantic_same[len(semantic_same) // 2:] > 0)
            total += len(semantic_same)

        self.info("evaluate with {}, acc: {}, total: {}, acc/total: {}".format(
            self.train_steps, acc, total, acc * 1. / total if total else 'Nan'))

        return acc, total

    def _str2ids(self, s: str) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s))

    def _snn_tj_transformation(
            self, tj: List[ActionMasterStr]
    ) -> Tuple[np.ndarray, np.ndarray]:
        master_ids = np.zeros(
            (self.hp.num_turns, self.hp.num_tokens), dtype=np.float32)
        action_ids = np.zeros(
            (self.hp.num_turns, self.hp.n_tokens_per_action), dtype=np.float32)

        for i, s in enumerate([x.action for x in tj]):
            s_ids = self._str2ids(s)
            s_len = min(len(s_ids), self.hp.n_tokens_per_action)
            action_ids[i, :s_len] = s_ids[:s_len]

        for i, s in enumerate([x.master for x in tj]):
            s_ids = self._str2ids(s)
            s_len = min(len(s_ids), self.hp.num_tokens)
            master_ids[i, :s_len] = s_ids[:s_len]

        return action_ids, master_ids

    @classmethod
    def get_snn_pairs(
            cls,
            hash_states2tjs: Dict[str, Dict[int, List[int]]],
            tjs: Trajectory,
            size: int
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]],
               List[Tuple[int, int]]]:

        non_empty_keys = list(
            filter(lambda x: hash_states2tjs[x] != {},
                   hash_states2tjs.keys()))
        perm_keys = list(np.random.permutation(non_empty_keys))
        state_pairs = list(zip(non_empty_keys, perm_keys))

        target_set = []
        same_set = []
        diff_set = []

        i = 0
        while i < size:
            for j, (sk1, sk2) in enumerate(state_pairs):
                if sk1 == sk2:
                    sk2 = non_empty_keys[(j + 1) % len(non_empty_keys)]

                try:
                    tid_pair = np.random.choice(
                        list(hash_states2tjs[sk1].keys()),
                        size=2, replace=False)
                except ValueError:
                    tid_pair = list(hash_states2tjs[sk1].keys()) * 2

                target_tid = tid_pair[0]
                same_tid = tid_pair[1]

                if (target_tid == same_tid and
                        len(hash_states2tjs[sk1][same_tid]) == 1):
                    continue

                diff_tid = np.random.choice(
                    list(hash_states2tjs[sk2].keys()), size=None)

                # remove empty trajectory
                if (target_tid not in tjs.trajectories
                        or same_tid not in tjs.trajectories
                        or diff_tid not in tjs.trajectories):
                    continue

                target_sid = np.random.choice(
                    list(hash_states2tjs[sk1][target_tid]), size=None)
                same_sid = np.random.choice(
                    list(hash_states2tjs[sk1][same_tid]), size=None)
                diff_sid = np.random.choice(
                    list(hash_states2tjs[sk2][diff_tid]), size=None)

                target_set.append((target_tid, target_sid))
                same_set.append((same_tid, same_sid))
                diff_set.append((diff_tid, diff_sid))

                i += 1
                if i >= size:
                    break

        return target_set, same_set, diff_set

    def get_snn_tjs(
            self, tjs: Trajectory, tid_sid_set: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray, List[List[ActionMasterStr]]]:
        trajectories = [
            tjs.fetch_state_by_idx(tid, sid) for tid, sid in tid_sid_set]

        action_ids, master_ids = zip(*[
            self._snn_tj_transformation(tj) for tj in trajectories])

        return action_ids, master_ids, trajectories
