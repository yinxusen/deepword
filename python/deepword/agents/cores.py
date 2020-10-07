import math
from abc import ABC
from copy import deepcopy
from os import path
from typing import Dict
from typing import Optional, List, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Session
from tensorflow.contrib.training import HParams
from tensorflow.python.client import device_lib
from tensorflow.summary import FileWriter
from tensorflow.train import Saver
from termcolor import colored

from deepword.agents.utils import ActionMaster, ObsInventory
from deepword.agents.utils import GenSummary
from deepword.agents.utils import batch_drrn_action_input
from deepword.agents.utils import bert_commonsense_input
from deepword.agents.utils import get_action_idx_pair
from deepword.agents.utils import get_best_1d_q
from deepword.agents.utils import get_best_batch_ids
from deepword.agents.utils import get_path_tags
from deepword.agents.utils import id_real2batch
from deepword.log import Logging
from deepword.models.dqn_model import DQNModel
from deepword.models.export_models import CommonsenseModel
from deepword.models.export_models import DRRNModel
from deepword.models.export_models import VecDRRNModel
from deepword.models.export_models import DSQNModel
from deepword.models.export_models import DSQNZorkModel
from deepword.models.export_models import GenDQNModel
from deepword.tokenizers import init_tokens
from deepword.utils import ctime, report_status
from deepword.utils import eprint
from deepword.utils import flatten
from deepword.utils import get_hash
from deepword.utils import model_name2clazz


class BaseCore(Logging, ABC):
    """
    Core: used for agents to compute policy.
    Core objects are isolated with games and gaming platforms. They work with
    agents, receiving trajectories, actions, and then compute a policy for
    agents.

    How to get trajectories, actions, and how to choose actions given policies
    are decided by agents.
    """
    def __init__(
            self, hp: HParams, model_dir: str) -> None:
        """
        Initialize A Core for an agent.

        Args:
            hp: hyper-parameters, see :py:mod:`deepword.hparams`
            model_dir: path to save or load model
        """
        super(BaseCore, self).__init__()
        self.hp = hp
        self.model_dir = model_dir
        self.loaded_ckpt_step: int = 0
        self.is_training: bool = True

    def policy(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            action_mask: np.ndarray) -> np.ndarray:
        """
        Infer from policy.

        Args:
            trajectory: a list of ActionMaster
            state: the current game state of observation + inventory
            action_matrix: a matrix of all actions for the game, 2D array,
             each row represents a tokenized and indexed action.
            action_len: 1D array, length for each action.
            action_mask: 1D array, indices of admissible actions from
             all actions of the game.

        Returns:
            Q-values for actions in the action_matrix
        """
        raise NotImplementedError()

    def save_model(self, t: Optional[int] = None) -> None:
        """
        Save current model with training steps

        Args:
            t: training steps, `None` falls back to default global steps
        """
        raise NotImplementedError()

    def init(
            self, is_training: bool, load_best: bool = False,
            restore_from: Optional[str] = None) -> None:
        """
        Initialize models of the core.

        Args:
            is_training: training or evaluation
            load_best: load from best weights, otherwise last weights
            restore_from: path to restore
        """
        raise NotImplementedError()

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int, others: Any) -> np.ndarray:
        """
        Train the core with one batch of data.

        Args:
            pre_trajectories: previous trajectories
            post_trajectories: post trajectories
            pre_states: previous states
            post_states: post states
            action_matrix: all actions for each of previous trajectories
            action_len: length of actions
            pre_action_mask: action masks for each of previous trajectories
            post_action_mask: action masks for each of post trajectories
            dones: game terminated or not for post trajectories
            rewards: rewards received for reaching post trajectories
            action_idx: actions used for reaching post trajectories
            b_weight: 1D array, weight for each data point
            step: current training step
            others: other information passed for training purpose

        Returns: Absolute loss between expected Q-value and predicted Q-value
         for each data point
        """
        raise NotImplementedError()

    def create_or_reload_target_model(
            self, restore_from: Optional[str] = None) -> None:
        """
        Create (if not exist) or reload weights for the target model

        Args:
            restore_from: the path to restore weights
        """
        raise NotImplementedError()


class TFCore(BaseCore, ABC):
    """
    Agent core implemented through Tensorflow.
    """
    def __init__(
            self, hp: HParams, model_dir: str) -> None:
        """
        Args:
            hp: hyper-parameters
            model_dir: path to model dir
        """
        super(TFCore, self).__init__(hp, model_dir)
        self.hp: HParams = hp
        self.model_dir: str = model_dir
        self.model: Optional[DQNModel] = None
        self.target_model: Optional[DQNModel] = None
        self.sess: Optional[Session] = None
        self.target_sess: Optional[Session] = None
        self.train_summary_writer: Optional[FileWriter] = None
        self.ckpt_path = path.join(self.model_dir, 'last_weights')
        self.best_ckpt_path = path.join(self.model_dir, 'best_weights')
        self.ckpt_prefix = path.join(self.ckpt_path, 'after-epoch')
        self.best_ckpt_prefix = path.join(self.best_ckpt_path, 'after-epoch')
        self.saver: Optional[Saver] = None
        self.target_saver: Optional[Saver] = None
        self.d4train, self.d4eval, self.d4target = self._init_devices()

    @classmethod
    def _init_devices(cls):
        all_devices = device_lib.list_local_devices()
        eprint("number of all devices: {}".format(len(all_devices)))
        eprint(all_devices)
        devices = [d.name for d in all_devices if d.device_type == "GPU"]
        eprint("list of GPU devices:")
        eprint(devices)
        eprint("number of gpu devices: {}".format(len(devices)))
        if len(devices) == 0:
            d4train, d4eval, d4target = (
                "/device:CPU:0", "/device:CPU:0", "/device:CPU:0")
        elif len(devices) == 1:
            d4train, d4eval, d4target = devices[0], devices[0], devices[0]
        elif len(devices) == 2:
            d4train = devices[0]
            d4eval, d4target = devices[1], devices[1]
        else:
            d4train = devices[0]
            d4eval = devices[1]
            d4target = devices[2]
        return d4train, d4eval, d4target

    def save_model(self, t: Optional[int] = None) -> None:
        """
        Save model to model_dir with the number of training steps.

        Args:
            t: number of training steps, `None` falls back to global step
        """
        self.info('save model')
        if t is None:
            t = tf.train.get_or_create_global_step(graph=self.model.graph)
        self.saver.save(self.sess, self.ckpt_prefix, global_step=t)

    def _get_target_model(self) -> Tuple[Any, Session]:
        if self.target_model is None:
            target_model = self.model
            target_sess = self.sess
        else:
            target_model = self.target_model
            target_sess = self.target_sess
        return target_model, target_sess

    def safe_loading(
            self, model: DQNModel, sess: Session, saver: Saver,
            restore_from: str) -> int:
        """
        Load weights from restore_from to model.
        If weights in loaded model are incompatible with current model,
        try to load those weights that have the same name.

        This method is useful when saved model lacks of training part, e.g.
        Adam optimizer.

        Args:
            model: A tensorflow model
            sess: A tensorflow session
            saver: A tensorflow saver
            restore_from: the path to restore the model

        Returns:
            training steps
        """
        self.info(
            colored(
                "Try to restore parameters from: {}".format(restore_from),
                "magenta", attrs=["bold", "underline"]))
        with model.graph.as_default():
            try:
                saver.restore(sess, restore_from)
            except Exception as e:
                self.debug(
                    "Restoring from saver failed,"
                    " try to restore from safe saver\n{}".format(e))
                all_saved_vars = list(
                    map(lambda v: v[0],
                        tf.train.list_variables(restore_from)))
                self.debug(
                    "Try to restore with safe saver with vars:\n{}".format(
                        "\n".join(all_saved_vars)))
                all_vars = tf.global_variables()
                self.debug("all vars:\n{}".format(
                    "\n".join([v.op.name for v in all_vars])))
                var_list = [v for v in all_vars if v.op.name in all_saved_vars]
                self.debug("Matched vars:\n{}".format(
                    "\n".join([v.name for v in var_list])))
                safe_saver = tf.train.Saver(var_list=var_list)
                safe_saver.restore(sess, restore_from)
            global_step = tf.train.get_or_create_global_step()
            trained_steps = sess.run(global_step)
        return trained_steps

    def _create_model_instance(self, device):
        model_creator = model_name2clazz(self.hp.model_creator)
        self.debug(
            "try to create train model: {}".format(self.hp.model_creator))
        return model_creator.get_train_model(self.hp, device)

    def _create_eval_model_instance(self, device):
        model_creator = model_name2clazz(self.hp.model_creator)
        self.debug(
            "try to create eval model: {}".format(self.hp.model_creator))
        return model_creator.get_eval_model(self.hp, device)

    def _create_model(
            self, is_training=True,
            device: Optional[str] = None) -> Tuple[Session, Any, Saver]:
        if is_training:
            device = device if device else self.d4train
            model = self._create_model_instance(device)
            self.info("create train model on device {}".format(device))
        else:
            device = device if device else self.d4eval
            model = self._create_eval_model_instance(device)
            self.info("create eval model on device {}".format(device))

        conf = tf.ConfigProto(
            log_device_placement=False, allow_soft_placement=True)
        sess = tf.Session(graph=model.graph, config=conf)
        with model.graph.as_default():
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(
                max_to_keep=self.hp.max_snapshot_to_keep,
                save_relative_paths=True)
        return sess, model, saver

    def set_d4eval(self, device: str) -> None:
        """
        Set the device for evaluation, e.g. "/device:CPU:0", "/device:GPU:1"
        Otherwise, a default device allocation will be used.

        Args:
            device: device name
        """
        self.d4eval = device

    def _load_model(
            self, sess: Session, model: DQNModel, saver: Saver,
            restore_from: Optional[str] = None, load_best=False) -> int:
        if restore_from is None:
            if load_best:
                restore_from = tf.train.latest_checkpoint(self.best_ckpt_path)
            else:
                restore_from = tf.train.latest_checkpoint(self.ckpt_path)

        if restore_from is not None:
            trained_step = self.safe_loading(model, sess, saver, restore_from)
        else:
            self.warning(colored(
                "No checkpoint to load, using untrained model",
                "red", "on_white", ["bold", "blink", "underline"]))
            trained_step = 0
        return trained_step

    def trajectory2input(
            self, trajectory: List[ActionMaster]) -> Tuple[List[int], int]:
        """
        generate src, src_len from trajectory, trimmed by hp.num_tokens

        Args:
            trajectory: List of ActionMaster

        Returns:
            src: source indices
            src_len: length of the src
        """

        tj = flatten([x.ids for x in trajectory])
        tj = tj if len(tj) <= self.hp.num_tokens else tj[-self.hp.num_tokens:]
        return tj, len(tj)

    def batch_trajectory2input(
            self, trajectories: List[List[ActionMaster]]
    ) -> Tuple[List[List[int]], List[int]]:
        """
        generate batch of src, src_len, trimmed by hp.num_tokens

        see :py:func:`deepword.agents.cores.TFCore.trajectory2input`

        Args:
            trajectories: a batch of trajectories

        Returns:
            batch of src
            batch of src_len
        """
        tjs, lens = zip(*[self.trajectory2input(x) for x in trajectories])

        # notice that `trajectory2input` has already trimmed by hp.num_tokens
        # so no len will be larger than hp.num_tokens
        max_len = max(lens)
        tjs = [
            x + [0] * (max_len - len(x)) if len(x) < max_len else x
            for x in tjs]
        return tjs, lens

    def init(
            self, is_training: bool, load_best: bool = False,
            restore_from: Optional[str] = None) -> None:
        """
        Initialize the core.

        1. create the model
        2. load the model if there are saved models
        3. create target model for training

        Args:
            is_training: True for training, False for evaluation
            load_best: load best model, otherwise load last weights
            restore_from: specify the load path, `load_best` will be disabled
        """
        self.is_training = is_training
        if self.model is None:
            self.sess, self.model, self.saver = self._create_model(
                self.is_training)
        self.loaded_ckpt_step = self._load_model(
            self.sess, self.model, self.saver, restore_from, load_best)

        if self.is_training:
            if self.loaded_ckpt_step > 0:
                self.create_or_reload_target_model(restore_from)
            train_summary_dir = path.join(
                self.model_dir, "summaries", "train")
            self.train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, self.sess.graph)

    def save_best_model(self) -> None:
        """
        Save current model to the best weights dir
        """
        self.info("save the best model so far")
        self.saver.save(
            self.sess, self.best_ckpt_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=self.model.graph))
        self.info("the best model saved")

    def create_or_reload_target_model(
            self, restore_from: Optional[str] = None) -> None:
        """
        Create the target model if not exists, then load model from the most
        recent saved weights.

        Args:
            restore_from: path to load target model, `None` falls back to
             default.
        """
        if self.target_sess is None:
            self.debug("create target model ...")
            (self.target_sess, self.target_model, self.target_saver
             ) = self._create_model(is_training=False, device=self.d4target)
        else:
            pass
        trained_step = self._load_model(
            self.target_sess, self.target_model, self.target_saver,
            restore_from, load_best=False)
        self.debug(
            "load target model from trained step {}".format(trained_step))


class BertCore(TFCore):
    """
    The agent that explores commonsense ability of BERT models.
    This agent combines each trajectory with all its actions together, separated
    with [SEP] in the middle. Then feeds the sentence into BERT to get a score
    from the [CLS] token.
    refer to https://arxiv.org/pdf/1810.04805.pdf for fine-tuning and evaluation
    """
    def __init__(self, hp, model_dir):
        super(BertCore, self).__init__(hp, model_dir)
        self.model: Optional[CommonsenseModel] = None
        self.target_model: Optional[CommonsenseModel] = None

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int,
            others: Any) -> np.ndarray:
        raise NotImplementedError("BertCore doesn't support for training")

    def policy(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            action_mask: np.ndarray) -> np.ndarray:
        action_matrix = action_matrix[action_mask, :]
        action_len = action_len[action_mask]
        src, src_len = self.trajectory2input(trajectory)
        inp, seg_tj_action, inp_size = bert_commonsense_input(
            action_matrix, action_len, src, src_len,
            self.hp.sep_val_id, self.hp.cls_val_id, self.hp.num_tokens)
        n_actions = inp.shape[0]
        self.debug("number of actions: {}".format(n_actions))
        # TODO: better allowed batch
        allowed_batch_size = 32
        n_batches = int(math.ceil(n_actions * 1. / allowed_batch_size))
        self.debug("compute q-values through {} batches".format(n_batches))
        total_q_actions = []
        for i in range(n_batches):
            ss = i * allowed_batch_size
            ee = min((i + 1) * allowed_batch_size, n_actions)
            q_actions_t = self.sess.run(self.model.q_actions, feed_dict={
                self.model.src_: inp[ss: ee],
                self.model.seg_tj_action_: seg_tj_action[ss: ee],
                self.model.src_len_: inp_size[ss: ee],
            })
            total_q_actions.append(q_actions_t)

        q_actions = np.concatenate(total_q_actions, axis=-1)
        return q_actions


class DQNCore(TFCore):
    """
    DQNAgent that treats actions as types
    """

    def __init__(self, hp, model_dir):
        super(DQNCore, self).__init__(hp, model_dir)

    def policy(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            action_mask: np.ndarray) -> np.ndarray:
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        """
        src, src_len = self.trajectory2input(trajectory)
        q_actions = self.sess.run(self.model.q_actions, feed_dict={
            self.model.src_: [src],
            self.model.src_len_: [src_len]
        })[0]
        return q_actions[action_mask]

    def _compute_expected_q(
            self,
            action_mask: List[np.ndarray],
            trajectories: List[List[ActionMaster]],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:
        """
        Compute expected q values given post trajectories and post actions

        notice that action_mask, tids, sids should belong to post game states,
        while dones, rewards belong to pre game states.
        """

        src, src_len = self.batch_trajectory2input(trajectories)
        target_model, target_sess = self._get_target_model()
        # target network provides the value used as expected q-values
        qs_target = target_sess.run(
            target_model.q_actions,
            feed_dict={
                target_model.src_: src,
                target_model.src_len_: src_len})

        # current network decides which action provides best q-value
        qs_dqn = self.sess.run(
            self.model.q_actions,
            feed_dict={
                self.model.src_: src,
                self.model.src_len_: src_len})

        expected_q = np.zeros_like(rewards)
        for i in range(len(expected_q)):
            expected_q[i] = rewards[i]
            if not dones[i]:
                action_idx, _ = get_best_1d_q(qs_dqn[i, action_mask[i]])
                real_action_idx = action_mask[i][action_idx]
                expected_q[i] += (
                        self.hp.gamma * qs_target[i, real_action_idx])
        return expected_q

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int, others: Any) -> np.ndarray:

        expected_q = self._compute_expected_q(
            action_mask=post_action_mask, trajectories=post_trajectories,
            dones=dones, rewards=rewards)

        pre_src, pre_src_len = self.batch_trajectory2input(pre_trajectories)
        _, summaries, loss_eval, abs_loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={
                self.model.src_: pre_src,
                self.model.src_len_: pre_src_len,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: action_idx,
                self.model.expected_q_: expected_q})

        self.info('loss: {}'.format(loss_eval))
        self.train_summary_writer.add_summary(
            summaries, step - self.hp.observation_t)
        return abs_loss


class TabularCore(BaseCore):
    """
    Tabular-wise DQN agent that uses matrix to store q-vectors and uses
    hashed values of observation + inventory as game states
    """
    def __init__(self, hp, model_dir):
        super(TabularCore, self).__init__(hp, model_dir)
        self.hp = hp
        self.q_mat_prefix = "q_mat"
        # model of tabular Q-learning, map from state to q-vectors
        self.q_mat: Dict[str, np.ndarray] = dict()
        self.target_q_mat: Dict[str, np.ndarray] = dict()
        self.state2hash: Dict[ObsInventory, str] = dict()
        self.model_dir = model_dir
        self.ckpt_prefix = "after-epoch"
        self.ckpt_path = path.join(self.model_dir, self.q_mat_prefix)

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int, others: Any) -> np.ndarray:

        expected_q = self._compute_expected_q(
            post_action_mask, post_states, dones, rewards)

        pre_hash_states = [
            self.get_state_hash(state[0]) for state in pre_states]

        abs_loss = np.zeros_like(rewards)
        for i, ps in enumerate(pre_hash_states):
            if ps not in self.q_mat:
                self.q_mat[ps] = np.zeros(self.hp.n_actions)
            prev_q_val = self.q_mat[ps][action_idx[i]]
            delta_q_val = expected_q[i] - prev_q_val
            abs_loss[i] = abs(delta_q_val)
            self.q_mat[ps][action_idx[i]] = (
                    prev_q_val + delta_q_val * b_weight[i])
        return abs_loss

    def policy(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            action_mask: np.ndarray) -> np.ndarray:
        hs = self.get_state_hash(state)
        q_actions = self.q_mat.get(hs, np.zeros(self.hp.n_actions))
        return q_actions[action_mask]

    def create_or_reload_target_model(
            self, restore_from: Optional[str] = None) -> None:
        self.target_q_mat = deepcopy(self.q_mat)

    def init(
            self, is_training: bool, load_best: bool = False,
            restore_from: Optional[str] = None) -> None:
        self.is_training = is_training
        try:
            if not restore_from:
                tags = get_path_tags(
                    self.ckpt_path, self.ckpt_prefix)
                self.loaded_ckpt_step = max(tags)
                restore_from = path.join(
                    self.ckpt_path, "{}-{}.npz".format(
                        self.ckpt_prefix, self.loaded_ckpt_step))
            else:
                # TODO: fetch loaded ckpt step
                pass
            npz_q_mat = np.load(restore_from, allow_pickle=True)
            q_mat_key = npz_q_mat["q_mat_key"]
            q_mat_val = npz_q_mat["q_mat_val"]
            self.q_mat = dict(zip(q_mat_key, q_mat_val))
            self.debug("load q_mat from file")
            self.target_q_mat = deepcopy(self.q_mat)
            self.debug("init target_q_mat with q_mat")
        except IOError as e:
            self.debug("load q_mat error:\n{}".format(e))
        pass

    def save_model(self, t: Optional[int] = None) -> None:
        q_mat_path = path.join(
            self.ckpt_path, "{}-{}.npz".format(self.ckpt_prefix, t))
        np.savez(
            q_mat_path,
            q_mat_key=list(self.q_mat.keys()),
            q_mat_val=list(self.q_mat.values()))

    def get_state_hash(self, state: ObsInventory) -> str:
        if state in self.state2hash:
            hs = self.state2hash[state]
        else:
            hs = get_hash(state.obs + "\n" + state.inventory)
            self.state2hash[state] = hs
        return hs

    def _compute_expected_q(
            self,
            action_mask: List[np.ndarray],
            states: List[ObsInventory],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:
        post_hash_states = [self.get_state_hash(state) for state in states]
        post_qs_target = np.asarray(
            [self.target_q_mat.get(s, np.zeros(self.hp.n_actions))
             for s in post_hash_states])
        post_qs_dqn = np.asarray(
            [self.q_mat.get(s, np.zeros(self.hp.n_actions))
             for s in post_hash_states])

        expected_q = np.zeros_like(rewards)
        for i in range(len(expected_q)):
            expected_q[i] = rewards[i]
            if not dones[i]:
                action_idx, _ = get_best_1d_q(post_qs_dqn[i, action_mask[i]])
                real_action_idx = action_mask[i][action_idx]
                expected_q[i] += (
                        self.hp.gamma * post_qs_target[i, real_action_idx])
        return expected_q


class DRRNCore(TFCore):
    """
    DRRN agent that treats actions as meaningful sentences
    """

    def __init__(self, hp, model_dir):
        super(DRRNCore, self).__init__(hp, model_dir)
        self.model: Optional[DRRNModel] = None
        self.target_model: Optional[DRRNModel] = None

    def policy(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            action_mask: np.ndarray) -> np.ndarray:
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        """
        admissible_action_matrix = action_matrix[action_mask, :]
        admissible_action_len = action_len[action_mask]
        actions_repeats = [len(action_mask)]

        src, src_len = self.trajectory2input(trajectory)
        q_actions = self.sess.run(self.model.q_actions, feed_dict={
            self.model.src_: [src],
            self.model.src_len_: [src_len],
            self.model.actions_: admissible_action_matrix,
            self.model.actions_len_: admissible_action_len,
            self.model.actions_repeats_: actions_repeats
        })

        return q_actions

    def _compute_expected_q(
            self,
            action_mask: List[np.ndarray],
            trajectories: List[List[ActionMaster]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:

        post_src, post_src_len = self.batch_trajectory2input(trajectories)
        actions, actions_lens, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)

        target_model, target_sess = self._get_target_model()
        post_qs_target = target_sess.run(
            target_model.q_actions,
            feed_dict={
                target_model.src_: post_src,
                target_model.src_len_: post_src_len,
                target_model.actions_: actions,
                target_model.actions_len_: actions_lens,
                target_model.actions_repeats_: actions_repeats})

        post_qs_dqn = self.sess.run(
            self.model.q_actions,
            feed_dict={
                self.model.src_: post_src,
                self.model.src_len_: post_src_len,
                self.model.actions_: actions,
                self.model.actions_len_: actions_lens,
                self.model.actions_repeats_: actions_repeats})

        best_actions_idx = get_best_batch_ids(post_qs_dqn, actions_repeats)
        best_qs = post_qs_target[best_actions_idx]
        expected_q = (
                np.asarray(rewards) +
                np.asarray(dones) * self.hp.gamma * best_qs)
        return expected_q

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int,
            others: Any) -> np.ndarray:

        t1 = ctime()
        expected_q = self._compute_expected_q(
            post_action_mask, post_trajectories, action_matrix, action_len,
            dones, rewards)
        t1_end = ctime()

        t2 = ctime()
        pre_src, pre_src_len = self.batch_trajectory2input(pre_trajectories)
        t2_end = ctime()

        t3 = ctime()
        (actions, actions_lens, actions_repeats, id_real2mask
         ) = batch_drrn_action_input(
            action_matrix, action_len, pre_action_mask)
        action_batch_ids = id_real2batch(
            action_idx, id_real2mask, actions_repeats)
        t3_end = ctime()

        t4 = ctime()
        _, summaries, loss_eval, abs_loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={
                self.model.src_: pre_src,
                self.model.src_len_: pre_src_len,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: action_batch_ids,
                self.model.expected_q_: expected_q,
                self.model.actions_: actions,
                self.model.actions_len_: actions_lens,
                self.model.actions_repeats_: actions_repeats})
        t4_end = ctime()

        self.info(report_status([
            ("t1", t1_end - t1),
            ("t2", t2_end - t2),
            ("t3", t3_end - t3),
            ("t4", t4_end - t4)
        ]))

        self.train_summary_writer.add_summary(
            summaries, step - self.hp.observation_t)
        return abs_loss


class DSQNZorkCore(DQNCore):
    def __init__(self, hp, model_dir):
        super(DQNCore, self).__init__(hp, model_dir)
        self.model: Optional[DSQNZorkModel] = None
        self.target_model: Optional[DSQNZorkModel] = None

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int,
            others: Any) -> np.ndarray:

        src, src_len, src2, src2_len, labels = others.get()

        expected_q = self._compute_expected_q(
            action_mask=post_action_mask, trajectories=post_trajectories,
            dones=dones, rewards=rewards)

        pre_src, pre_src_len = self.batch_trajectory2input(pre_trajectories)

        _, summaries, weighted_loss, abs_loss = self.sess.run(
            [self.model.merged_train_op, self.model.weighted_train_summary_op,
             self.model.weighted_loss, self.model.abs_loss],
            feed_dict={
                self.model.src_: pre_src,
                self.model.src_len_: pre_src_len,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: action_idx,
                self.model.expected_q_: expected_q,
                self.model.snn_src_: src,
                self.model.snn_src_len_: src_len,
                self.model.snn_src2_: src2,
                self.model.snn_src2_len_: src2_len,
                self.model.labels_: labels})
        self.train_summary_writer.add_summary(
            summaries, step - self.hp.observation_t)
        return abs_loss

    # TODO: refine the code
    def eval_snn(
            self,
            snn_data: Tuple[
                np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            batch_size: int = 32) -> float:

        src, src_len, src2, src2_len, labels = snn_data
        eval_data_size = len(src)
        self.info("start eval with size {}".format(eval_data_size))
        n_iter = (eval_data_size // batch_size) + 1
        total_acc = 0
        total_samples = 0
        for i in range(n_iter):
            self.debug("eval snn iter {} total {}".format(i, n_iter))
            non_empty_src = list(filter(
                lambda x: x[1][0] != 0 and x[1][1] != 0,
                enumerate(zip(src_len, src2_len))))
            non_empty_src_idx = [x[0] for x in non_empty_src]
            src = src[non_empty_src_idx, :]
            src_len = src_len[non_empty_src_idx]
            src2 = src2[non_empty_src_idx, :]
            src2_len = src2_len[non_empty_src_idx]
            labels = labels[non_empty_src_idx]
            labels = labels.astype(np.int32)
            pred, diff_two_states = self.sess.run(
                [self.model.semantic_same, self.model.h_states_diff],
                feed_dict={self.model.snn_src_: src,
                           self.model.snn_src2_: src2,
                           self.model.snn_src_len_: src_len,
                           self.model.snn_src2_len_: src2_len})
            pred_labels = (pred > 0).astype(np.int32)
            total_acc += np.sum(np.equal(labels, pred_labels))
            total_samples += len(src)
        if total_samples == 0:
            avg_acc = -1
        else:
            avg_acc = total_acc * 1. / total_samples
            self.debug("valid sample size {}".format(total_samples))
        return avg_acc
    pass


class DSQNCore(DRRNCore):
    def __init__(self, hp, model_dir):
        super(DRRNCore, self).__init__(hp, model_dir)
        self.model: Optional[DSQNModel] = None
        self.target_model: Optional[DSQNModel] = None

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int,
            others: Any) -> np.ndarray:

        t0 = ctime()
        src, src_len, src2, src2_len, labels = others.get()
        t0_end = ctime()

        t1 = ctime()
        expected_q = self._compute_expected_q(
            post_action_mask, post_trajectories, action_matrix, action_len,
            dones, rewards)
        t1_end = ctime()

        t2 = ctime()
        pre_src, pre_src_len = self.batch_trajectory2input(pre_trajectories)
        t2_end = ctime()

        t3 = ctime()
        (actions, actions_lens, actions_repeats, id_real2mask
         ) = batch_drrn_action_input(
            action_matrix, action_len, pre_action_mask)
        action_batch_ids = id_real2batch(
            action_idx, id_real2mask, actions_repeats)
        t3_end = ctime()

        t4 = ctime()
        _, summaries, weighted_loss, abs_loss = self.sess.run(
            [self.model.merged_train_op, self.model.weighted_train_summary_op,
             self.model.weighted_loss, self.model.abs_loss],
            feed_dict={
                self.model.src_: pre_src,
                self.model.src_len_: pre_src_len,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: action_batch_ids,
                self.model.expected_q_: expected_q,
                self.model.actions_: actions,
                self.model.actions_len_: actions_lens,
                self.model.actions_repeats_: actions_repeats,
                self.model.snn_src_: src,
                self.model.snn_src_len_: src_len,
                self.model.snn_src2_: src2,
                self.model.snn_src2_len_: src2_len,
                self.model.labels_: labels})
        t4_end = ctime()
        self.info(report_status([
            ("t0", t0_end - t0),
            ("t1", t1_end - t1),
            ("t2", t2_end - t2),
            ("t3", t3_end - t3),
            ("t4", t4_end - t4)
        ]))
        self.train_summary_writer.add_summary(
            summaries, step - self.hp.observation_t)
        return abs_loss

    # TODO: refine the code
    def eval_snn(
            self,
            snn_data: Tuple[
                np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            batch_size: int = 32) -> float:

        src, src_len, src2, src2_len, labels = snn_data
        eval_data_size = len(src)
        self.info("start eval with size {}".format(eval_data_size))
        n_iter = (eval_data_size // batch_size) + 1
        total_acc = 0
        total_samples = 0
        for i in range(n_iter):
            self.debug("eval snn iter {} total {}".format(i, n_iter))
            non_empty_src = list(filter(
                lambda x: x[1][0] != 0 and x[1][1] != 0,
                enumerate(zip(src_len, src2_len))))
            non_empty_src_idx = [x[0] for x in non_empty_src]
            src = src[non_empty_src_idx, :]
            src_len = src_len[non_empty_src_idx]
            src2 = src2[non_empty_src_idx, :]
            src2_len = src2_len[non_empty_src_idx]
            labels = labels[non_empty_src_idx]
            labels = labels.astype(np.int32)
            pred, diff_two_states = self.sess.run(
                [self.model.semantic_same, self.model.h_states_diff],
                feed_dict={self.model.snn_src_: src,
                           self.model.snn_src2_: src2,
                           self.model.snn_src_len_: src_len,
                           self.model.snn_src2_len_: src2_len})
            pred_labels = (pred > 0).astype(np.int32)
            total_acc += np.sum(np.equal(labels, pred_labels))
            total_samples += len(src)
        if total_samples == 0:
            avg_acc = -1
        else:
            avg_acc = total_acc * 1. / total_samples
            self.debug("valid sample size {}".format(total_samples))
        return avg_acc


class GenDQNCore(TFCore):
    def __init__(self, hp, model_dir):
        super(GenDQNCore, self).__init__(hp, model_dir)
        self.model: Optional[GenDQNModel] = None
        self.target_model: Optional[GenDQNModel] = None
        self.hp, self.tokenizer = init_tokens(hp)

    def summary(
            self, token_idx: np.ndarray, col_eos_idx: np.ndarray,
            p_gen: np.ndarray, sum_logits: np.ndarray
    ) -> List[GenSummary]:
        res_summary = []
        for i in range(len(token_idx)):
            n_cols = col_eos_idx[i]
            ids = list(token_idx[i, :n_cols])
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            gens = list(p_gen[i, :n_cols])
            log_prob = sum_logits[i] / n_cols
            res_summary.append(GenSummary(ids, tokens, gens, log_prob, n_cols))
        return res_summary

    def decode_action(
            self, trajectory: List[ActionMaster]) -> GenSummary:
        self.debug("trajectory: {}".format(trajectory))
        src, src_len = self.trajectory2input(trajectory)
        self.debug("src: {}".format(src))
        self.debug("src_len: {}".format(src_len))
        beam_size = 1
        temperature = 1
        use_greedy = True

        self.debug("use_greedy: {}, temperature: {}".format(
            use_greedy, temperature))

        res = self.sess.run(
            [self.model.decoded_idx_infer,
             self.model.col_eos_idx,
             self.model.p_gen_infer,
             self.model.decoded_logits_infer],
            feed_dict={
                self.model.src_: [src],
                self.model.src_len_: [src_len],
                self.model.temperature_: temperature,
                self.model.beam_size_: beam_size,
                self.model.use_greedy_: use_greedy
            })

        token_idx = res[0]
        col_eos_idx = res[1]
        p_gen = res[2]
        decoded_logits = res[3]
        res_summary = self.summary(
            token_idx, col_eos_idx, p_gen, decoded_logits)
        res_summary = flatten(
            [sorted(res_summary[i: i + beam_size], key=lambda x: -x.log_prob)
             for i in range(0, len(res_summary), beam_size)]
        )

        self.debug("generated actions:\n{}".format(
            "\n".join([str(x) for x in res_summary])))

        return res_summary[0]

    def policy(
            self, trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            action_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError("GenDQNCore doesn't support policy")

    def _compute_expected_q(
            self,
            trajectories: List[List[ActionMaster]],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:
        """
        Compute expected q values given post trajectories and post actions
        """

        src, src_len = self.batch_trajectory2input(trajectories)
        target_model, target_sess = self._get_target_model()
        # target network provides the value used as expected q-values
        qs_target = target_sess.run(
            target_model.decoded_logits_infer,
            feed_dict={
                target_model.src_: src,
                target_model.src_len_: src_len,
                target_model.beam_size_: 1,
                target_model.use_greedy_: True,
                target_model.temperature_: 1.
            })

        # current network decides which action provides best q-value
        s_argmax_q, valid_len = self.sess.run(
            [self.model.decoded_idx_infer, self.model.col_eos_idx],
            feed_dict={
                self.model.src_: src,
                self.model.src_len_: src_len,
                self.model.beam_size_: 1,
                self.model.use_greedy_: True,
                self.model.temperature_: 1.})

        expected_q = np.zeros_like(rewards)
        for i in range(len(expected_q)):
            expected_q[i] = rewards[i]
            if not dones[i]:
                expected_q[i] += self.hp.gamma * np.mean(
                    qs_target[i, range(valid_len[i]),
                              s_argmax_q[i, :valid_len[i]]])

        return expected_q

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int,
            others: Any) -> np.ndarray:

        expected_q = self._compute_expected_q(
            trajectories=post_trajectories, dones=dones, rewards=rewards)

        src, src_len = self.batch_trajectory2input(pre_trajectories)
        action_token_ids = others
        action_id_in, action_id_out, new_action_len = get_action_idx_pair(
            np.asarray(action_token_ids), np.asarray(action_len),
            self.hp.sos_id, self.hp.eos_id)
        self.debug("action in/out example:\n{} -- {}\n{} -- {}".format(
            action_id_in[0, :],
            self.tokenizer.de_tokenize(action_id_in[0, :]),
            action_id_out[0, :],
            self.tokenizer.de_tokenize(action_id_out[0, :])))

        _, summaries, loss_eval, abs_loss = self.sess.run(
            [self.model.train_op,
             self.model.train_summary_op,
             self.model.loss,
             self.model.abs_loss],
            feed_dict={
                self.model.src_: src,
                self.model.src_len_: src_len,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: action_id_in,
                self.model.action_idx_out_: action_id_out,
                self.model.action_len_: new_action_len,
                self.model.expected_q_: expected_q})
        self.train_summary_writer.add_summary(
            summaries, step - self.hp.observation_t)
        return abs_loss


class PGNCore(TFCore):
    """Generate admissible actions for games, given only trajectory"""
    def __init__(self, hp, model_dir):
        super(PGNCore, self).__init__(hp, model_dir)
        self.model: Optional[GenDQNModel] = None
        self.hp, self.tokenizer = init_tokens(hp)

    def summary(
            self, action_idx: np.ndarray, col_eos_idx: np.ndarray,
            decoded_logits: np.ndarray, p_gen: np.ndarray, beam_size: int
    ) -> List[GenSummary]:
        """
        Return [ids, tokens, generation probabilities of each token, q_action]
        sorted by q_action (from larger to smaller)
        q_action: the average of decoded logits of selected tokens
        """
        res_summary = []
        for bid in range(beam_size):
            n_cols = col_eos_idx[bid]
            ids = list(action_idx[bid, :n_cols])
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            gen_prob_per_token = list(p_gen[bid, :n_cols])
            q_action = np.sum(decoded_logits[bid, :n_cols, ids]) / n_cols
            res_summary.append(
                GenSummary(ids, tokens, gen_prob_per_token, q_action, n_cols))
        res_summary = list(reversed(sorted(res_summary, key=lambda x: x[-1])))
        return res_summary

    def decode(
            self, trajectory: List[ActionMaster], beam_size: int,
            temperature: float, use_greedy: bool) -> List[GenSummary]:
        src, src_len = self.trajectory2input(trajectory)
        res = self.sess.run(
            [self.model.decoded_idx_infer,
             self.model.col_eos_idx,
             self.model.decoded_logits_infer,
             self.model.p_gen_infer],
            feed_dict={
                self.model.src_: [src],
                self.model.src_len_: [src_len],
                self.model.temperature_: temperature,
                self.model.beam_size_: beam_size,
                self.model.use_greedy_: use_greedy
            })
        action_idx = res[0]
        col_eos_idx = res[1]
        decoded_logits = res[2]
        p_gen = res[3]
        res_summary = self.summary(
            action_idx, col_eos_idx, decoded_logits, p_gen, beam_size)
        self.debug("generated results:\n{}".format(
            "\n".join([str(x) for x in res_summary])))
        return res_summary

    def generate_admissible_actions(
            self, trajectory: List[ActionMaster]) -> List[str]:

        if self.hp.decode_concat_action:
            res = self.decode(
                trajectory, beam_size=1, temperature=1., use_greedy=False)
            concat_actions = self.tokenizer.de_tokenize(res[0].ids)
            actions = [a.strip() for a in concat_actions.split(";")]
        else:
            res = self.decode(
                trajectory, beam_size=20, temperature=1., use_greedy=False)
            actions = [self.tokenizer.de_tokenize(x.ids) for x in res]
        return actions

    def policy(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            action_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PGNCore doesn't support policy")

    def train_one_batch(
            self, pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray], dones: List[bool],
            rewards: List[float], action_idx: List[int],
            b_weight: np.ndarray, step: int,
            others: Any) -> np.ndarray:
        raise NotImplementedError("PGNCore doesn't support train")


class FastCore(TFCore):
    def __init__(self, hp, model_dir):
        super(FastCore, self).__init__(hp, model_dir)
        self.model: Optional[VecDRRNModel] = None
        self.target_model: Optional[VecDRRNModel] = None
        self._str2vec: Dict[Tuple[Any], np.ndarray] = dict()

    def pad_or_trim(self, src: List[List[int]]) -> List[List[int]]:
        max_len = min(max([len(x) for x in src]), self.hp.num_tokens)
        return [
            x + [self.hp.padding_val_id] * (max_len - len(x))
            if len(x) < max_len else x[:max_len] for x in src]

    def get_sentence_embeddings(
            self, src: List[List[int]]) -> List[np.ndarray]:
        embeddings = self.sess.run(
            self.model.sentence_embeddings,
            feed_dict={self.model.src_: self.pad_or_trim(src)})
        return list(embeddings)

    def get_action_embeddings(
            self, src: List[np.ndarray]) -> List[np.ndarray]:
        embeddings = self.sess.run(
            self.model.sentence_embeddings,
            feed_dict={self.model.src_: src})
        return list(embeddings)

    def trajectory2vector(self, trajectory: List[ActionMaster]) -> np.ndarray:
        strings = flatten([(am.action_ids, am.master_ids) for am in trajectory])
        unknown = [x for x in strings if tuple(x) not in self._str2vec]
        if unknown:
            self.debug(
                "get sentence embeddings for {} masters and actions".format(
                    len(unknown)))
            embeddings = self.get_sentence_embeddings(unknown)
            self._str2vec.update(
                dict(zip([tuple(x) for x in unknown], embeddings)))
        vectors = [self._str2vec[tuple(x)] for x in strings]
        return np.sum(vectors, axis=0)

    def actions2vectors(self, actions: np.ndarray) -> np.ndarray:
        assert actions.ndim == 2, "actions should have dim-2"
        unknown = [x for x in actions if tuple(x) not in self._str2vec]
        if unknown:
            self.debug(
                "get sentence embeddings for {} action(s)".format(
                    len(unknown)))
            embeddings = self.get_action_embeddings(unknown)
            self._str2vec.update(
                dict(zip([tuple(x) for x in unknown], embeddings)))
        vectors = np.asarray([self._str2vec[tuple(x)] for x in actions])
        return vectors

    def policy(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray,
            action_mask: np.ndarray) -> np.ndarray:
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        """
        admissible_action_matrix = action_matrix[action_mask, :]
        actions_repeats = [len(action_mask)]

        vec_src = self.trajectory2vector(trajectory)
        vec_actions = self.actions2vectors(admissible_action_matrix)
        q_actions = self.sess.run(self.model.q_actions, feed_dict={
            self.model.vec_src_: [vec_src],
            self.model.vec_actions_: vec_actions,
            self.model.actions_repeats_: actions_repeats,
            self.model.state_id_: [state.sid]
        })

        return q_actions

    def _compute_expected_q(
            self,
            action_mask: List[np.ndarray],
            trajectories: List[List[ActionMaster]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            dones: List[bool],
            rewards: List[float]) -> np.ndarray:

        post_vec_src = [self.trajectory2vector(tj) for tj in trajectories]
        actions, actions_lens, actions_repeats, _ = batch_drrn_action_input(
            action_matrix, action_len, action_mask)
        vec_actions = self.actions2vectors(actions)

        target_model, target_sess = self._get_target_model()
        post_qs_target = target_sess.run(
            target_model.q_actions,
            feed_dict={
                target_model.vec_src_: post_vec_src,
                target_model.vec_actions_: vec_actions,
                target_model.actions_repeats_: actions_repeats})

        post_qs_dqn = self.sess.run(
            self.model.q_actions,
            feed_dict={
                self.model.vec_src_: post_vec_src,
                self.model.vec_actions_: vec_actions,
                self.model.actions_repeats_: actions_repeats})

        best_actions_idx = get_best_batch_ids(post_qs_dqn, actions_repeats)
        best_qs = post_qs_target[best_actions_idx]
        expected_q = (
                np.asarray(rewards) +
                np.asarray(dones) * self.hp.gamma * best_qs)
        return expected_q

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: List[np.ndarray],
            post_action_mask: List[np.ndarray],
            dones: List[bool],
            rewards: List[float],
            action_idx: List[int],
            b_weight: np.ndarray,
            step: int,
            others: Any) -> np.ndarray:

        t1 = ctime()
        expected_q = self._compute_expected_q(
            post_action_mask, post_trajectories, action_matrix, action_len,
            dones, rewards)
        t1_end = ctime()

        t2 = ctime()
        pre_vec_src = [self.trajectory2vector(tj) for tj in pre_trajectories]
        t2_end = ctime()

        t3 = ctime()
        (actions, actions_lens, actions_repeats, id_real2mask
         ) = batch_drrn_action_input(
            action_matrix, action_len, pre_action_mask)
        action_batch_ids = id_real2batch(
            action_idx, id_real2mask, actions_repeats)
        vec_actions = self.actions2vectors(actions)
        t3_end = ctime()

        t4 = ctime()
        _, summaries, loss_eval, abs_loss = self.sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={
                self.model.vec_src_: pre_vec_src,
                self.model.b_weight_: b_weight,
                self.model.action_idx_: action_batch_ids,
                self.model.expected_q_: expected_q,
                self.model.vec_actions_: vec_actions,
                self.model.actions_repeats_: actions_repeats})
        t4_end = ctime()

        self.info(report_status([
            ("t1", t1_end - t1),
            ("t2", t2_end - t2),
            ("t3", t3_end - t3),
            ("t4", t4_end - t4)
        ]))

        self.train_summary_writer.add_summary(
            summaries, step - self.hp.observation_t)
        return abs_loss
