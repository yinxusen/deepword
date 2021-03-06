import math
import random
import sys
import time
import traceback
from os import path
from queue import Queue
from threading import Thread
from typing import Tuple, List, Union, Any, Optional, Dict, Generator

import numpy as np
import tensorflow as tf
from deeptextworld.agents.base_agent import DRRNMemoTeacher
from tensorflow import Session
from tensorflow.contrib.training import HParams
from tensorflow.summary import FileWriter
from tensorflow.train import Saver
from termcolor import colored
from tqdm import trange, tqdm

from deepword.action import ActionCollector
from deepword.agents.utils import ActionMaster
from deepword.agents.utils import Memolet
from deepword.agents.utils import batch_drrn_action_input
from deepword.agents.utils import bert_nlu_input
from deepword.agents.utils import get_action_idx_pair
from deepword.agents.utils import get_best_batch_ids
from deepword.agents.utils import get_path_tags
from deepword.agents.utils import sample_batch_ids
from deepword.hparams import save_hparams, output_hparams
from deepword.log import Logging
from deepword.students.utils import batch_dqn_input, align_batch_str
from deepword.tokenizers import init_tokens
from deepword.trajectory import Trajectory
from deepword.utils import flatten, softmax
from deepword.utils import load_uniq_lines
from deepword.utils import model_name2clazz, bytes2idx


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def set(self, key, val):
        self.__dict__[key] = val

    def get(self, key):
        return self.__dict__[key]


class StudentLearner(Logging):
    def __init__(
            self, hp: HParams, model_dir: str, train_data_dir: Optional[str],
            eval_data_path: Optional[str] = None) -> None:
        super(StudentLearner, self).__init__()
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
        self.info(output_hparams(self.hp))

        self.sess = None
        self.model = None
        self.saver = None
        self.sw = None
        self.train_steps = None
        self.queue = None

        # filter allowed gids from memory during training
        # if allowed_gids is empty, use all memory
        # allowed_gids.txt:
        # game name [TAB] game ID
        self.allowed_gids = set()
        fn_allowed_gids = path.join(self.model_dir, "allowed_gids.txt")
        if path.isfile(fn_allowed_gids):
            self.allowed_gids = set(
                [x.split("\t")[1] for x in load_uniq_lines(fn_allowed_gids)])

    def _get_compatible_snapshot_tag(self, data_dir: str) -> List[int]:
        action_tags = get_path_tags(data_dir, self.action_prefix)
        memo_tags = get_path_tags(data_dir, self.memo_prefix)
        tjs_tags = get_path_tags(data_dir, self.tjs_prefix)
        hs2tj_tags = get_path_tags(data_dir, self.hs2tj_prefix)

        valid_tags = set(action_tags)
        valid_tags.intersection_update(memo_tags)
        valid_tags.intersection_update(tjs_tags)
        valid_tags.intersection_update(hs2tj_tags)

        return list(valid_tags)

    def _get_combined_data_path(
            self, data_dir: str) -> List[Tuple[str, str, str, str]]:
        valid_tags = self._get_compatible_snapshot_tag(data_dir)
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
            self, device_placement: str, training: bool,
            restore_from: Optional[str] = None,
    ) -> Tuple[Session, Any, Saver, int]:
        """
        create and load model from restore_from
        if restore_from is None, use the latest checkpoint from last_weights
        if model_dir
        """
        model_clazz = model_name2clazz(self.hp.model_creator)
        if training:
            model = model_clazz.get_train_student_model(
                hp=self.hp,
                device_placement=device_placement)
        else:
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

            if restore_from is None:
                restore_from = tf.train.latest_checkpoint(self.load_from)

            if restore_from is not None:
                trained_steps = model.safe_loading(sess, saver, restore_from)
            else:
                self.warning(colored(
                    "No checkpoint to load, using untrained model",
                    "red", "on_white", ["bold", "blink", "underline"]))
                trained_steps = 0

        return sess, model, saver, trained_steps

    @classmethod
    def lst_str2am(
            cls, tj: List[str], allow_unfinished_tj: bool = False
    ) -> List[ActionMaster]:
        tj = [""] + tj

        if not allow_unfinished_tj and len(tj) % 2 != 0:
            raise ValueError("wrong old trajectory: {}".format(tj))

        res_tj = []
        i = 0
        while i < len(tj) // 2:
            res_tj.append(
                ActionMaster(
                    action=tj[i * 2], master=tj[i * 2 + 1],
                    action_ids=[], master_ids=[], objective_ids=[]))
            i += 1

        return res_tj

    @classmethod
    def tjs_str2am(
            cls, old_tjs: Trajectory[str]) -> Trajectory[ActionMaster]:
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

    @classmethod
    def hs2tj_old2new(
            cls, old_hs2tj: Dict[str, Dict[int, List[int]]]
    ) -> Dict[str, Dict[int, List[int]]]:
        """
        sid need to be halved.
        Args:
            old_hs2tj:

        Returns:

        """
        new_hs2tj = dict()
        for sk in old_hs2tj:
            new_hs2tj[sk] = dict()
            for tid in old_hs2tj[sk]:
                new_hs2tj[sk][tid] = list(np.asarray(old_hs2tj[sk][tid]) // 2)
        return new_hs2tj

    def _load_snapshot(
            self, memo_path: str, tjs_path: str, action_path: str,
            hs2tj_path: str
    ) -> Tuple[List[Memolet], Trajectory[ActionMaster], ActionCollector,
               Dict[str, Dict[int, List[int]]]]:
        memory = np.load(memo_path, allow_pickle=True)["data"]
        if isinstance(memory[0], DRRNMemoTeacher):
            self.warning("load old data with DRRNMemoTeacher")
            return self._load_snapshot_v1(
                memo_path, tjs_path, action_path, hs2tj_path)
        elif isinstance(memory[0], Memolet):
            self.warning("load new data with Memolet")
            return self._load_snapshot_v2(
                memo_path, tjs_path, action_path, hs2tj_path)
        else:
            raise ValueError(
                "Unrecognized memory type: {}".format(type(memory[0])))

    def _load_snapshot_v1(
            self, memo_path: str, tjs_path: str, action_path: str,
            hs2tj_path: str
    ) -> Tuple[List[Memolet], Trajectory[ActionMaster], ActionCollector,
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
        hash_states2tjs = self.hs2tj_old2new(hs2tj["hs2tj"][0])

        return memory, tjs, actions, hash_states2tjs

    def _load_snapshot_v2(
            self, memo_path: str, tjs_path: str, action_path: str,
            hs2tj_path: str
    ) -> Tuple[List[Memolet], Trajectory[ActionMaster], ActionCollector,
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
        self.info("try to add batch data: {}".format(combined_data_path))
        while True:
            if training and append_new_data:
                new_combined_data_path = self._get_combined_data_path(
                    self.train_data_dir)
                if set(new_combined_data_path) != set(combined_data_path):
                    self.info(
                        "update training data: {}".format(combined_data_path))
                    combined_data_path = new_combined_data_path
            for tp, ap, mp, hsp in sorted(
                    combined_data_path, key=lambda k: random.random()):
                memory, tjs, action_collector, _ = self._load_snapshot(
                    mp, tp, ap, hsp)
                if training:
                    if not self.allowed_gids:
                        random.shuffle(memory)
                    else:
                        self.info(
                            "before gid filtering: {}".format(len(memory)))
                        memory = [
                            x for x in memory if x.gid in self.allowed_gids]
                        self.info("after gid filtering: {}".format(len(memory)))
                        random.shuffle(memory)
                else:
                    memory = memory[:5000]
                i = 0
                while i < int(math.ceil(len(memory) * 1. / self.hp.batch_size)):
                    ss = i * self.hp.batch_size
                    ee = min((i + 1) * self.hp.batch_size, len(memory))
                    batch_memory = memory[ss:ee]
                    try:
                        queue.put(self._prepare_data(
                            batch_memory, tjs, action_collector),
                        )
                    except Exception as e:
                        self.error("add_batch error: {}".format(e))
                        traceback.print_tb(e.__traceback__)
                        raise RuntimeError()
                    i += 1
            # only load data once if not training
            if not training:
                break
        return

    def _prepare_data(
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

    def _prepare_training(
            self
    ) -> Tuple[Session, Any, Saver, FileWriter, int, Queue]:
        sess, model, saver, train_steps = self._prepare_model(
            "/device:GPU:0", training=True)

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
            self.info("waiting data ... (retry times: {})".format(wait_times))
            time.sleep(10)
            wait_times -= 1

        if self.queue.empty():
            self.warning("No data received. exit")
            return

        epoch_size = self.hp.save_gap_t

        self.info("start training")
        data_in_queue = True
        for et in trange(n_epochs, ascii=True, desc="epoch"):
            for it in trange(epoch_size, ascii=True, desc="step"):
                try:
                    data = self.queue.get(timeout=1000)
                    self._train_impl(
                        data, self.train_steps + et * epoch_size + it)
                except Exception as e:
                    data_in_queue = False
                    self.info("no more data: {}".format(e))
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(
                        exc_type, exc_value, exc_traceback, limit=None,
                        file=sys.stdout)
                    break
            self.saver.save(
                self.sess, self.ckpt_prefix,
                global_step=tf.train.get_or_create_global_step(
                    graph=self.model.graph))
            self.info("finish and save {} epoch".format(et))
            if not data_in_queue:
                break
        return

    def _train_impl(self, data: Tuple, train_step: int) -> None:
        """
        Train the model one time given data.

        :param data:
        :param train_step:
        :return:
        """
        raise NotImplementedError()

    def _prepare_test(
            self, device_placement: str = "/device:GPU:0",
            restore_from: Optional[str] = None
    ) -> Tuple[Session, Any, Saver, int, Queue]:
        sess, model, saver, train_steps = self._prepare_model(
            device_placement, training=False, restore_from=restore_from)
        queue = Queue()
        return sess, model, saver, train_steps, queue

    # TODO: fix the OOM problem
    def preprocess_input(self, data_dir):
        valid_tags = self._get_compatible_snapshot_tag(data_dir)
        queue = []

        for tag in valid_tags:
            tp = path.join(
                data_dir, "{}-{}.npz".format(self.tjs_prefix, tag))
            hsp = path.join(
                data_dir, "{}-{}.npz".format(self.hs2tj_prefix, tag))
            ap = path.join(
                data_dir,
                "{}-{}.npz".format(self.action_prefix, tag))
            mp = path.join(
                data_dir, "{}-{}.npz".format(self.memo_prefix, tag))

            memory, tjs, action_collector, hash_states2tjs = \
                self._load_snapshot(mp, tp, ap, hsp)

            i = 0
            while i < int(math.ceil(len(memory) * 1. / self.hp.batch_size)):
                if i % 100 == 0:
                    self.info("process batch - {}".format(i))
                ss = i * self.hp.batch_size
                ee = min((i + 1) * self.hp.batch_size, len(memory))
                batch_memory = memory[ss:ee]
                queue.append(self._prepare_data(
                    batch_memory, tjs, action_collector),
                )
                i += 1

            np.savez(
                "{}/student-data-{}.npz".format(self.model_dir, tag),
                data=queue)

    # TODO: data loader to use batch size
    def data_loader(
            self, data_path: str, batch_size: int, training: bool
    ) -> Generator[Tuple, None, None]:
        data_tags = sorted(get_path_tags(data_path, prefix="student-data"))
        self.info("load snn tags: {}".format(data_tags))
        while True:
            for tag in tqdm(sorted(data_tags, key=lambda k: random.random())):
                self.info("load data from {}".format(tag))
                data = np.load(
                    path.join(data_path, "student-data-{}.npz".format(tag)),
                    allow_pickle=True)
                data = data["data"]
                for x in data:
                    yield x
            # only load data one time for evaluation
            if not training:
                self.info("snn data loader finished")
                break

    def _test_impl(self, data: Tuple) -> np.ndarray:
        """
        Test the model one time given data.

        :param data:
        :return:
        """
        raise NotImplementedError()

    def test(
            self, device_placement: str = "/device:GPU:0",
            restore_from: Optional[str] = None) -> Tuple[float, int]:

        if self.sess is None:
            (self.sess, self.model, self.saver, self.train_steps,
             self.queue) = self._prepare_test(device_placement, restore_from)

        data_loader = self.data_loader(
            data_path=self.eval_data_path, batch_size=self.hp.batch_size,
            training=False)

        self.info("start test")
        total_test = 0
        total_acc = []
        # TODO: control the evaluation size, current we use 625 batches
        # TODO: with 8 trajectories (32 samples) per batch
        i = 0
        for data in data_loader:
            if i >= 625:
                break
            acc = self._test_impl(data)
            if i % 100 == 0:
                self.debug(
                    "progress: {}, current acc: {}, data: {}".format(
                        i, np.mean(acc), len(acc)))
            total_test += len(acc)
            total_acc.append(acc)
            i += 1

        acc = float(np.mean(total_acc))
        self.info("step: {}, acc: {}, total: {}".format(
            self.train_steps, acc, total_test))
        return acc, total_test


class DRRNLearner(StudentLearner):
    def __init__(
            self, hp: HParams, model_dir: str, train_data_dir: str) -> None:
        super(DRRNLearner, self).__init__(hp, model_dir, train_data_dir)

    def _train_impl(self, data, train_step):
        (p_states, p_len, actions, action_len, actions_repeats, expected_qs
         ) = data
        _, summaries, loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss],
            feed_dict={
                self.model.src_: p_states,
                self.model.src_len_: p_len,
                self.model.actions_: actions,
                self.model.actions_len_: action_len,
                self.model.actions_repeats_: actions_repeats,
                self.model.action_idx_: np.arange(len(expected_qs)),
                self.model.expected_q_: expected_qs,
                self.model.b_weight_: [1.]})
        self.debug("\nloss: {}".format(loss))
        self.sw.add_summary(summaries, train_step)

    def _test_impl(self, data: Tuple) -> np.ndarray:
        raise NotImplementedError()

    def _prepare_data(self, b_memory, tjs, action_collector):
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        expected_qs = flatten([list(m.q_actions) for m in b_memory])

        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, _ = batch_dqn_input(
            states, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id,
            with_action_padding=self.hp.action_padding_in_tj,
            max_action_size=self.hp.n_tokens_per_action
        )
        action_len = (
            [action_collector.get_action_len(gid) for gid in game_id])
        action_matrix = (
            [action_collector.get_action_matrix(gid) for gid in game_id])
        actions, action_len, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)
        return (
            p_states, p_len, actions, action_len, actions_repeats, expected_qs)


class GenLearner(StudentLearner):
    def _train_impl(self, data, train_step):
        (p_states, p_len, master_mask, actions_in, actions_out, action_len,
         b_weight) = data
        _, summaries, loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss],
            feed_dict={
                self.model.src_: p_states,
                self.model.src_len_: p_len,
                self.model.src_seg_: master_mask,
                self.model.action_idx_: actions_in,
                self.model.action_idx_out_: actions_out,
                self.model.action_len_: action_len,
                self.model.b_weight_: b_weight})
        self.sw.add_summary(summaries, train_step)
        self.debug("\nloss: {}".format(loss))

    def _test_impl(self, data: Tuple) -> np.ndarray:
        raise NotImplementedError()

    def _prepare_data(self, b_memory, tjs, action_collector):
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        expected_qs = np.asarray(flatten([list(m.q_actions) for m in b_memory]))
        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, master_mask = batch_dqn_input(
            states, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id,
            with_action_padding=self.hp.action_padding_in_tj,
            max_action_size=self.hp.n_tokens_per_action)
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
        b_weight = [1.]
        return (
            p_states, p_len, master_mask, actions_in, actions_out, action_len,
            b_weight)


class GenMixActionsLearner(GenLearner):
    def _test_impl(self, data: Tuple) -> np.ndarray:
        raise NotImplementedError()

    def _prepare_data(self, b_memory, tjs, action_collector):
        n_classes = 4
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        # b_weight is softmax(q-values)
        # TODO: since every trajectory has different size of admissible actions,
        #   the softmax version of q-values could have different range for each
        #   trajectory. But it's fine for now, since we only need the weight
        #   inside admissible actions for each trajectory.
        if self.hp.gen_loss_weighted_by_qs:
            b_weight = np.asarray(flatten(
                [list(softmax(m.q_actions)) for m in b_memory]))
        else:
            b_weight = np.asarray(flatten(
                [list(softmax(np.zeros_like(m.q_actions)))
                 for m in b_memory]))

        action_len = (
            [action_collector.get_action_len(gid) for gid in game_id])
        action_matrix = (
            [action_collector.get_action_matrix(gid) for gid in game_id])
        actions, action_len, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)

        batch_q_idx = sample_batch_ids(
            b_weight, actions_repeats, k=n_classes)
        selected_b_weights = b_weight[batch_q_idx]

        actions_in, actions_out, action_len = get_action_idx_pair(
            actions[batch_q_idx], action_len[batch_q_idx],
            self.hp.sos_id, self.hp.eos_id)

        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, master_mask = batch_dqn_input(
            states, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id,
            with_action_padding=self.hp.action_padding_in_tj,
            max_action_size=self.hp.n_tokens_per_action)
        p_states = np.repeat(p_states, n_classes, axis=0)
        p_len = np.repeat(p_len, n_classes, axis=0)
        master_mask = np.repeat(master_mask, n_classes, axis=0)

        return (
            p_states, p_len, master_mask, actions_in, actions_out, action_len,
            selected_b_weights[:, None])


class GenConcatActionsLearner(GenLearner):
    """
    TODO: we choose action_mask in memory; however, when all admissible actions
      are required, e.g. in pre-training without using q-values, we can use
      sys_action_mask.
      Notice that sys_action_mask won't compatible with q-values, since the
      q-values are computed against action_mask.
    """
    @classmethod
    def concat_actions(
            cls,
            action_ids: List[np.ndarray],
            action_len: List[int],
            action_weight: np.ndarray,
            sep_val_id: int,
            sort_by_weight: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """concat given action_ids into one output str"""
        if sort_by_weight:
            sorted_idx = np.argsort(-action_weight, axis=-1)
            action_ids = np.take_along_axis(action_ids, sorted_idx, axis=0)
            action_len = np.take_along_axis(action_len, sorted_idx)
            action_weight = np.take_along_axis(
                action_weight, sorted_idx, axis=-1)

        action_ids = flatten(
            [(x[:l], [sep_val_id]) for x, l in zip(action_ids, action_len)])
        concat_action = np.concatenate(action_ids, axis=0)[:-1]
        token_weight = np.repeat(action_weight, repeats=action_len + 1)[:-1]
        return concat_action, token_weight

    def _test_impl(self, data: Tuple) -> np.ndarray:
        raise NotImplementedError()

    def _prepare_data_v2(self, b_memory, tjs, action_collector):
        """prepare concat actions without using q-values to weigh"""
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        admissible_action_mask = [m.sys_action_mask for m in b_memory]

        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, master_mask = batch_dqn_input(
            states, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id,
            with_action_padding=self.hp.action_padding_in_tj,
            max_action_size=self.hp.n_tokens_per_action)

        actions = [
            " ; ".join(sorted(list(
                np.asarray(action_collector.get_actions(gid))[mid])))
            for gid, mid in zip(game_id, admissible_action_mask)]
        action_idx = [
            self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(concat_actions))
            for concat_actions in actions]

        action_matrix, action_len = align_batch_str(
            ids=action_idx, str_len_allowance=self.hp.max_decoding_size,
            padding_val_id=self.hp.padding_val_id,
            valid_len=[len(x) for x in action_idx])

        actions_in, actions_out, action_len = get_action_idx_pair(
            action_matrix, action_len, self.hp.sos_id, self.hp.eos_id)
        return (
            p_states, p_len, master_mask, actions_in, actions_out, action_len)

    def _prepare_data(self, b_memory, tjs, action_collector):
        """prepare concat action weighted by q-values"""
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        expected_qs = [m.q_actions for m in b_memory]

        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, master_mask = batch_dqn_input(
            states, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id,
            with_action_padding=self.hp.action_padding_in_tj,
            max_action_size=self.hp.n_tokens_per_action)

        batch_concat_action_in = []
        batch_concat_action_out = []
        batch_token_weight = []
        batch_concat_action_len = []

        for gid, mask, q_vals in zip(game_id, action_mask, expected_qs):
            action_len = action_collector.get_action_len(gid)[mask]
            action_ids = action_collector.get_action_matrix(gid)[mask]

            # step 1: concat actions
            concat_action, token_weight = self.concat_actions(
                action_ids, action_len, softmax(q_vals),
                self.tokenizer.vocab[";"], sort_by_weight=True)

            # step 2: padding or trimming to max allowance
            concat_len = len(concat_action)
            padding_size = self.hp.max_decoding_size - concat_len
            if padding_size > 0:
                concat_action = np.pad(
                    concat_action, (0, padding_size),
                    mode='constant', constant_values=self.hp.padding_val_id)
                token_weight = np.pad(
                    token_weight, (0, padding_size),
                    mode='constant', constant_values=0)
            else:
                concat_action = concat_action[:self.hp.max_decoding_size]
                token_weight = token_weight[:self.hp.max_decoding_size]
            concat_len = min(concat_len, self.hp.max_decoding_size)

            # step 3: create in/out format for PGN
            action_id_in = np.pad(
                concat_action, (1,),
                mode='constant', constant_values=self.hp.sos_id)
            # make sure original action_matrix is untouched.
            action_id_out = np.copy(concat_action)
            new_action_len = min(concat_len + 1, self.hp.max_decoding_size)
            action_id_out[new_action_len - 1] = self.hp.eos_id
            token_weight[new_action_len - 1] = token_weight[new_action_len - 2]

            # step 4: collect results
            batch_concat_action_in.append(action_id_in)
            batch_concat_action_out.append(action_id_out)
            batch_token_weight.append(token_weight)
            batch_concat_action_len.append(new_action_len)

        return (
            p_states, p_len, master_mask,
            np.asarray(batch_concat_action_in),
            np.asarray(batch_concat_action_out),
            np.asarray(batch_concat_action_len),
            np.asarray(batch_token_weight))


class NLULearner(StudentLearner):
    def _train_impl(self, data, train_step):
        inp, seg_tj_action, inp_len, selected_qs, swag_labels = data
        _, summaries = self.sess.run(
            [self.model.train_op, self.model.train_summary_op],
            feed_dict={
                self.model.src_: inp,
                self.model.src_len_: inp_len,
                self.model.seg_tj_action_: seg_tj_action,
                self.model.expected_q_: selected_qs
                })
        self.sw.add_summary(summaries, train_step)

    def _test_impl(self, data: Tuple) -> np.ndarray:
        inp, seg_tj_action, inp_len, selected_qs, swag_labels = data
        q_actions = self.sess.run(
            self.model.q_actions,
            feed_dict={
                self.model.src_: inp,
                self.model.src_len_: inp_len,
                self.model.seg_tj_action_: seg_tj_action
            })
        return np.abs(q_actions - selected_qs)

    def _prepare_data(self, b_memory, tjs, action_collector):
        n_classes = 4
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        expected_qs = np.asarray(flatten([list(m.q_actions) for m in b_memory]))

        action_len = (
            [action_collector.get_action_len(gid) for gid in game_id])
        action_matrix = (
            [action_collector.get_action_matrix(gid) for gid in game_id])
        actions, action_len, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)

        batch_q_idx = sample_batch_ids(
            expected_qs, actions_repeats, k=n_classes)
        selected_qs = expected_qs[batch_q_idx]

        # [CLS] + [trajectory] + [SEP] + [action] + [SEP]
        max_allowed_trajectory_size = (
            self.hp.num_tokens - 3 - self.hp.n_tokens_per_action)
        # fetch pre-trajectory
        states = tjs.fetch_batch_pre_states(trajectory_id, state_id)
        p_states, p_len, _ = batch_dqn_input(
            states, self.tokenizer, max_allowed_trajectory_size,
            self.hp.padding_val_id,
            with_action_padding=self.hp.action_padding_in_tj,
            max_action_size=self.hp.n_tokens_per_action)
        batch_size = len(p_states)

        action_len = action_len[batch_q_idx].reshape((batch_size, n_classes))
        actions = actions[batch_q_idx].reshape((batch_size, n_classes, -1))
        # all labels are zero, because the first one is always the best
        swag_labels = np.zeros((len(actions), ), dtype=np.int32)

        processed_input = [
            bert_nlu_input(
                am, al, tj, tj_len, self.hp.sep_val_id, self.hp.cls_val_id,
                self.hp.num_tokens)
            for am, al, tj, tj_len
            in zip(list(actions), list(action_len), p_states, p_len)]

        inp = np.concatenate([a[0] for a in processed_input], axis=0)
        seg_tj_action = np.concatenate([a[1] for a in processed_input], axis=0)
        inp_len = np.concatenate([a[2] for a in processed_input], axis=0)

        return inp, seg_tj_action, inp_len, selected_qs, swag_labels


class NLUClassificationLearner(NLULearner):
    def _train_impl(self, data, train_step):
        inp, seg_tj_action, inp_len, selected_qs, swag_labels = data
        _, summaries, loss = self.sess.run(
            [self.model.classification_train_op,
             self.model.classification_train_summary_op,
             self.model.classification_loss],
            feed_dict={
                self.model.src_: inp,
                self.model.src_len_: inp_len,
                self.model.seg_tj_action_: seg_tj_action,
                self.model.swag_labels_: swag_labels
                })
        self.sw.add_summary(summaries, train_step)
        self.debug("\nloss: {}".format(loss))

    def _test_impl(self, data: Tuple) -> np.ndarray:
        raise NotImplementedError()
