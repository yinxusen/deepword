import math
import random
import sys
import time
import traceback
from os import path
from queue import Queue
from threading import Thread
from typing import Tuple, List, Any, Optional, Dict

import numpy as np
import tensorflow as tf
from tensorflow import Session
from tensorflow.contrib.training import HParams
from tensorflow.summary import FileWriter
from tensorflow.train import Saver
from tqdm import trange

from deepword.action import ActionCollector
from deeptextworld.agents.base_agent import DRRNMemoTeacher
from deeptextworld.agents.utils import Memolet
from deepword.agents.utils import get_path_tags
from deepword.hparams import save_hparams, output_hparams
from deepword.students.utils import ActionMasterStr
from deepword.tokenizers import init_tokens
from deepword.trajectory import Trajectory
from deepword.utils import eprint, flatten
from deepword.utils import model_name2clazz, bytes2idx


class SentenceLearner(object):
    def __init__(
            self, hp: HParams, model_dir: str, train_data_dir: Optional[str],
            eval_data_path: Optional[str] = None) -> None:
        # prefix should match BaseAgent
        self.tjs_prefix = "trajectories"
        self.action_prefix = "actions"
        self.memo_prefix = "memo"
        self.hs2tj_prefix = "hs2tj"

        self.model_dir = model_dir
        self.train_data_dir = train_data_dir
        self.eval_data_path = eval_data_path

        self.load_from = path.join(self.model_dir, "last_weights")
        self.ckpt_prefix = path.join(self.load_from, "after-epoch")
        self.hp, self.tokenizer = init_tokens(hp)
        save_hparams(self.hp, path.join(model_dir, "hparams.json"))
        eprint(output_hparams(self.hp))

        self.sess = None
        self.model = None
        self.saver = None
        self.sw = None
        self.train_steps = None
        self.queue = None

    def _get_compatible_snapshot_tag(self) -> List[int]:
        action_tags = get_path_tags(self.train_data_dir, self.action_prefix)
        memo_tags = get_path_tags(self.train_data_dir, self.memo_prefix)
        tjs_tags = get_path_tags(self.train_data_dir, self.tjs_prefix)
        hs2tj_tags = get_path_tags(self.train_data_dir, self.hs2tj_prefix)

        valid_tags = set(action_tags)
        valid_tags.intersection_update(memo_tags)
        valid_tags.intersection_update(tjs_tags)
        valid_tags.intersection_update(hs2tj_tags)

        return list(valid_tags)

    def _get_combined_data_path(
            self, data_dir: str) -> List[Tuple[str, str, str, str]]:
        valid_tags = self._get_compatible_snapshot_tag()
        combined_data_path = []
        for tag in sorted(valid_tags, key=lambda k: random.random()):
            combined_data_path.append(
                (path.join(
                    data_dir, "{}-{}.npz".format(self.tjs_prefix, tag)),
                 path.join(
                     data_dir, "{}-{}.npz".format(self.action_prefix, tag)),
                 path.join(
                     data_dir, "{}-{}.npz".format(self.memo_prefix, tag)),
                 path.join(
                     data_dir, "{}-{}.npz".format(
                         self.hs2tj_prefix, tag))))

        return combined_data_path

    def _prepare_model(
            self, device_placement: str, restore_from: Optional[str] = None
    ) -> Tuple[Session, Any, Saver, int]:
        """
        create and load model from restore_from
        if restore_from is None, use the latest checkpoint from last_weights
        if model_dir
        """
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
            if restore_from is None:
                restore_from = tf.train.latest_checkpoint(self.load_from)
            saver.restore(sess, restore_from)
            trained_steps = sess.run(global_step)
            eprint("load student from ckpt: {}".format(restore_from))
        except Exception as e:
            eprint("load model failed: {}".format(e))
            trained_steps = 0
        return sess, model, saver, trained_steps

    def _prepare_eval_model(
            self, device_placement: str, restore_from: Optional[str] = None
    ) -> Tuple[Session, Any, Saver, int]:
        """
        create and load model from restore_from
        if restore_from is None, use the latest checkpoint from last_weights
        if model_dir
        """
        model_clazz = model_name2clazz(self.hp.model_creator)
        model = model_clazz.get_eval_student_model(
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
            if restore_from is None:
                restore_from = tf.train.latest_checkpoint(self.load_from)
            saver.restore(sess, restore_from)
            trained_steps = sess.run(global_step)
            eprint("load student from ckpt: {}".format(restore_from))
        except Exception as e:
            eprint("load model failed: {}".format(e))
            trained_steps = 0
        return sess, model, saver, trained_steps

    @classmethod
    def lst_str2am(
            cls, tj: List[str], allow_unfinished_tj: bool = False
    ) -> List[ActionMasterStr]:
        tj = [""] + tj

        if not allow_unfinished_tj and len(tj) % 2 != 0:
            raise ValueError("wrong old trajectory: {}".format(tj))

        res_tj = []
        i = 0
        while i < len(tj) // 2:
            res_tj.append(
                ActionMasterStr(action=tj[i * 2], master=tj[i * 2 + 1]))
            i += 1

        return res_tj

    @classmethod
    def tjs_str2am(
            cls, old_tjs: Trajectory[str]) -> Trajectory[ActionMasterStr]:
        tjs = Trajectory(num_turns=old_tjs.num_turns // 2, size_per_turn=1)
        tjs.curr_tj = cls.lst_str2am(old_tjs.curr_tj, allow_unfinished_tj=True)
        tjs.curr_tid = old_tjs.curr_tid
        tjs.trajectories = dict([
            (k, cls.lst_str2am(v)) for k, v in old_tjs.trajectories.items()])
        return tjs

    @classmethod
    def memo_old2new(cls, old_memo: List[DRRNMemoTeacher]) -> List[Memolet]:
        res = []
        for m in old_memo:
            mask = bytes2idx(m.action_mask, size=128)
            next_mask = bytes2idx(m.next_action_mask, size=128)
            res.append(Memolet(
                tid=m.tid, sid=m.sid // 2, gid=m.gid, aid=m.aid,
                token_id=None, a_len=None, a_type=None,
                reward=m.reward, is_terminal=m.is_terminal,
                action_mask=mask, sys_action_mask=None,
                next_action_mask=next_mask, next_sys_action_mask=None,
                q_actions=m.q_actions[mask]))
        return res

    def _load_snapshot(
            self, memo_path: str, tjs_path: str, action_path: str,
            hs2tj_path: str
    ) -> Tuple[List[Tuple], Trajectory[ActionMasterStr], ActionCollector,
               Dict[str, Dict[int, List[int]]]]:
        memory = np.load(memo_path, allow_pickle=True)["data"]
        if isinstance(memory[0], DRRNMemoTeacher):
            eprint("load old data with DRRNMemoTeacher")
            return self._load_snapshot_v1(
                memo_path, tjs_path, action_path, hs2tj_path)
        elif isinstance(memory[0], Memolet):
            eprint("load new data with Memolet")
            return self._load_snapshot_v2(
                memo_path, tjs_path, action_path, hs2tj_path)
        else:
            raise ValueError(
                "Unrecognized memory type: {}".format(type(memory[0])))

    def _load_snapshot_v1(
            self, memo_path: str, tjs_path: str, action_path: str,
            hs2tj_path: str
    ) -> Tuple[List[Tuple], Trajectory[ActionMasterStr], ActionCollector,
               Dict[str, Dict[int, List[int]]]]:
        """load snapshot for old data"""
        old_memory = np.load(memo_path, allow_pickle=True)["data"]
        old_memory = list(filter(
            lambda x: isinstance(x, DRRNMemoTeacher), old_memory))
        memory = self.memo_old2new(old_memory)

        old_tjs = Trajectory(
            num_turns=self.hp.num_turns * 2 + 1, size_per_turn=2)
        old_tjs.load_tjs(tjs_path)
        tjs = self.tjs_str2am(old_tjs)

        actions = ActionCollector(
            tokenizer=self.tokenizer,
            n_tokens=self.hp.n_tokens_per_action,
            unk_val_id=self.hp.unk_val_id,
            padding_val_id=self.hp.padding_val_id)
        actions.load_actions(action_path)

        hs2tj = np.load(hs2tj_path, allow_pickle=True)
        hash_states2tjs = hs2tj["hs2tj"][0]

        return memory, tjs, actions, hash_states2tjs

    def _load_snapshot_v2(
            self, memo_path: str, tjs_path: str, action_path: str,
            hs2tj_path: str
    ) -> Tuple[List[Tuple], Trajectory[ActionMasterStr], ActionCollector,
               Dict[str, Dict[int, List[int]]]]:
        memory = np.load(memo_path, allow_pickle=True)["data"]
        memory = list(filter(lambda x: isinstance(x, Memolet), memory))

        tjs = Trajectory(self.hp.num_turns)
        tjs.load_tjs(tjs_path)

        actions = ActionCollector(
            tokenizer=self.tokenizer,
            n_tokens=self.hp.n_tokens_per_action,
            unk_val_id=self.hp.unk_val_id,
            padding_val_id=self.hp.padding_val_id)
        actions.load_actions(action_path)

        hs2tj = np.load(hs2tj_path, allow_pickle=True)
        hash_states2tjs = hs2tj["hs2tj"][0]

        return memory, tjs, actions, hash_states2tjs

    def _add_batch(
            self, combined_data_path: List[Tuple[str, str, str]],
            queue: Queue, training: bool = True,
            append_new_data: bool = True) -> None:
        """
        :param combined_data_path:
        :param queue:
        :param training:
        :param append_new_data: scan train_data_dir for new data after every
            epoch of training.
        :return:
        """
        eprint("try to add batch data: {}".format(combined_data_path))
        while True:
            if training and append_new_data:
                new_combined_data_path = self._get_combined_data_path(
                    self.train_data_dir)
                if set(new_combined_data_path) != set(combined_data_path):
                    eprint(
                        "update training data: {}".format(combined_data_path))
                    combined_data_path = new_combined_data_path
            for tp, ap, mp, hsp in sorted(
                    combined_data_path, key=lambda k: random.random()):
                memory, tjs, action_collector, hash_states2tjs = \
                    self._load_snapshot(mp, tp, ap, hsp)

                # for every loaded snapshot, we sample SNN pairs
                # according to len(memory) / batch_size
                i = 0
                while i < int(math.ceil(len(memory) * 1. / self.hp.batch_size)):
                    data = self.get_snn_pairs(
                        hash_states2tjs=hash_states2tjs,
                        tjs=tjs,
                        batch_size=self.hp.batch_size)
                    try:
                        queue.put(data)
                    except Exception as e:
                        eprint("add_batch error: {}".format(e))
                        traceback.print_tb(e.__traceback__)
                        raise RuntimeError()
                    i += 1

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
            args=(self._get_combined_data_path(self.train_data_dir), queue))
        t.setDaemon(True)
        t.start()

        return sess, model, saver, sw, train_steps, queue

    def train(self, n_epochs: int) -> None:
        if self.sess is None:
            (self.sess, self.model, self.saver, self.sw, self.train_steps,
             self.queue) = self._prepare_training()

        wait_times = 10
        while wait_times > 0 and self.queue.empty():
            eprint("waiting data ... (retry times: {})".format(wait_times))
            time.sleep(10)
            wait_times -= 1

        if self.queue.empty():
            eprint("No data received. exit")
            return

        epoch_size = self.hp.save_gap_t

        eprint("start training")
        data_in_queue = True
        for et in trange(n_epochs, ascii=True, desc="epoch"):
            for it in trange(epoch_size, ascii=True, desc="step"):
                try:
                    data = self.queue.get(timeout=1000)
                    self._train_impl(
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

    def _train_impl(self, data: Tuple, train_step: int) -> None:
        target_set, same_set, diff_set = data
        _, summaries, loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss],
            feed_dict={
                self.model.target_set_: target_set,
                self.model.same_set_: same_set,
                self.model.diff_set_: diff_set})
        self.sw.add_summary(summaries, train_step)

    def _prepare_test(
            self, device_placement: str = "/device:GPU:0",
            restore_from: Optional[str] = None
    ) -> Tuple[Session, Any, Saver, int, Queue]:
        sess, model, saver, train_steps = self._prepare_eval_model(
            device_placement, restore_from)
        queue = Queue(maxsize=100)
        t = Thread(
            target=self._add_batch,
            args=(self._get_combined_data_path(self.train_data_dir),
                  queue, False))
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
            target_set, same_set, diff_set = data
            if i % 100 == 0:
                print("process a batch of {} .. {}".format(len(target_set), i))
                print("partial acc.: {}".format(
                    acc * 1. / total if total else "Nan"))

            semantic_same = self.sess.run(
                self.model.semantic_same,
                feed_dict={
                    self.model.target_set_: target_set,
                    self.model.same_set_: same_set,
                    self.model.diff_set_: diff_set})

            acc += np.count_nonzero(semantic_same[: len(semantic_same)//2] < 0)
            acc += np.count_nonzero(semantic_same[len(semantic_same)//2:] > 0)
            total += len(semantic_same)
            i += 1
        return acc, total

    def _str2ids(self, s: str) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s))

    def _snn_tj_transformation(self, tj: List[ActionMasterStr]) -> np.ndarray:
        ids = np.zeros(
            (self.hp.num_turns * 2, self.hp.num_tokens), dtype=np.float32)
        for i, s in enumerate(flatten([[x.action, x.master] for x in tj])):
            s_ids = self._str2ids(s)
            s_len = min(len(s_ids), self.hp.num_tokens)
            ids[i, :s_len] = s_ids[:s_len]
        return ids

    def get_snn_pairs(
            self,
            hash_states2tjs: Dict[str, Dict[int, List[int]]],
            tjs: Trajectory[ActionMasterStr],
            batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        non_empty_keys = list(
            filter(lambda x: hash_states2tjs[x] != {},
                   hash_states2tjs.keys()))

        hs_keys = np.random.choice(non_empty_keys, size=batch_size)
        diff_keys = [
            np.random.choice(
                list(filter(lambda x: x != k, non_empty_keys)), size=None)
            for k in hs_keys]

        target_tids = []
        same_tids = []
        for k in hs_keys:
            try:
                tid_pair = np.random.choice(
                    list(hash_states2tjs[k].keys()), size=2, replace=False)
            except ValueError:
                tid_pair = list(hash_states2tjs[k].keys()) * 2

            target_tids.append(tid_pair[0])
            same_tids.append(tid_pair[1])

        diff_tids = [np.random.choice(
            list(hash_states2tjs[k])) for k in diff_keys]

        target_sids = [
            np.random.choice(list(hash_states2tjs[k][tid]))
            for k, tid in zip(hs_keys, target_tids)]
        same_sids = [
            np.random.choice(list(hash_states2tjs[k][tid]))
            for k, tid in zip(hs_keys, same_tids)]
        diff_sids = [
            np.random.choice(list(hash_states2tjs[k][tid]))
            for k, tid in zip(diff_keys, diff_tids)]

        tgt_set = list(zip(target_tids, target_sids))
        same_set = list(zip(same_tids, same_sids))
        diff_set = list(zip(diff_tids, diff_sids))

        trajectories = [
            tjs.fetch_state_by_idx(tid, sid) for tid, sid in
            tgt_set + same_set + diff_set]

        batch_src = np.asarray(
            [self._snn_tj_transformation(tj) for tj in trajectories])
        tgt_src = batch_src[: len(tgt_set)]
        same_src = batch_src[len(tgt_set): len(tgt_set) + len(same_set)]
        diff_src = batch_src[-len(diff_set):]

        return tgt_src, same_src, diff_src

    def save_train_pairs(
            self, t: int, src: np.ndarray, src_len: np.ndarray,
            src2: np.ndarray, src2_len: np.ndarray, labels: np.ndarray) -> None:
        """
        Save SNN pairs for verification.

        Args:
            t: current training steps
            src: trajectories
            src_len: length of trajectories
            src2: paired trajectories
            src2_len: length of paired trajectories
            labels: `0` or `1` for same or different states
        """
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
