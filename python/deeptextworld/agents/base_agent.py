import glob
import os
import random
import re
from abc import ABC
from os import remove as prm
from os.path import join as pjoin
from typing import Any, Optional

import tensorflow as tf
from tensorflow import Session
from tensorflow.contrib.training import HParams
from tensorflow.python.client import device_lib
from tensorflow.summary import FileWriter
from tensorflow.train import Saver
from textworld import EnvInfos

from deeptextworld.action import ActionCollector
from deeptextworld.agents.utils import *
from deeptextworld.floor_plan import FloorPlanCollector
from deeptextworld.hparams import conventions
from deeptextworld.hparams import save_hparams, output_hparams, copy_hparams
from deeptextworld.models.dqn_model import DQNModel
from deeptextworld.trajectory import Trajectory
from deeptextworld.tree_memory import TreeMemory
from deeptextworld.utils import eprint
from deeptextworld.utils import model_name2clazz, get_hash, core_name2clazz


class BaseCore(Logging, ABC):
    def __init__(
            self, hp: HParams, model_dir: str, tokenizer: Tokenizer) -> None:
        super(BaseCore, self).__init__()
        self.hp = hp
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.loaded_ckpt_step: int = 0
        self.is_training: bool = True

    def get_a_policy_action(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray, actions: List[str],
            action_mask: np.ndarray,
            cnt_action: Optional[Dict[int, float]]) -> ActionDesc:
        raise NotImplementedError()

    def save_model(self, t: Optional[int] = None) -> None:
        raise NotImplementedError()

    def init(
            self, is_training: bool, load_best: bool = False,
            restore_from: Optional[str] = None) -> None:
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
        raise NotImplementedError()

    def create_or_reload_target_model(
            self, restore_from: Optional[str] = None) -> None:
        raise NotImplementedError()


class TFCore(BaseCore, ABC):
    def __init__(
            self, hp: HParams, model_dir: str, tokenizer: Tokenizer) -> None:
        super(TFCore, self).__init__(hp, model_dir, tokenizer)
        self.hp: HParams = hp
        self.model_dir: str = model_dir
        self.model: Optional[DQNModel] = None
        self.target_model: Optional[DQNModel] = None
        self.sess: Optional[Session] = None
        self.target_sess: Optional[Session] = None
        self.train_summary_writer: Optional[FileWriter] = None
        self.ckpt_path = os.path.join(self.model_dir, 'last_weights')
        self.best_ckpt_path = os.path.join(self.model_dir, 'best_weights')
        self.ckpt_prefix = os.path.join(self.ckpt_path, 'after-epoch')
        self.best_ckpt_prefix = os.path.join(self.best_ckpt_path, 'after-epoch')
        self.saver: Optional[Saver] = None
        self.target_saver: Optional[Saver] = None
        self.d4train, self.d4eval, self.d4target = self.init_devices()
        self.tokenizer = tokenizer

    @classmethod
    def init_devices(cls):
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
        self.info('save model')
        if t is None:
            t = tf.train.get_or_create_global_step(graph=self.model.graph)
        self.saver.save(self.sess, self.ckpt_prefix, global_step=t)

    def get_target_model(self) -> Tuple[Any, Session]:
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
        :param model:
        :param sess:
        :param saver:
        :param restore_from:
        :return: trained steps
        """
        self.info(
            "Try to restore parameters from: {}".format(restore_from))
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

    def create_model_instance(self, device):
        model_creator = model_name2clazz(self.hp.model_creator)
        self.debug(
            "try to create train model: {}".format(self.hp.model_creator))
        return model_creator.get_train_model(self.hp, device)

    def create_eval_model_instance(self, device):
        model_creator = model_name2clazz(self.hp.model_creator)
        self.debug(
            "try to create eval model: {}".format(self.hp.model_creator))
        return model_creator.get_eval_model(self.hp, device)

    def create_model(
            self, is_training=True,
            device: Optional[str] = None) -> Tuple[Session, Any, Saver]:
        if is_training:
            device = device if device else self.d4train
            model = self.create_model_instance(device)
            self.info("create train model on device {}".format(device))
        else:
            device = device if device else self.d4eval
            model = self.create_eval_model_instance(device)
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
        self.d4eval = device

    def load_model(
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
            self.warning('No checkpoint to load, using untrained model')
            trained_step = 0
        return trained_step

    def trajectory2input(
            self, trajectory: List[ActionMaster]
    ) -> Tuple[List[int], int, List[int]]:
        return dqn_input(
            trajectory, self.tokenizer, self.hp.num_tokens,
            self.hp.padding_val_id)

    def batch_trajectory2input(
            self, trajectories: List[List[ActionMaster]]
    ) -> Tuple[List[List[int]], List[int], List[List[int]]]:
        batch_src = []
        batch_src_len = []
        batch_mask = []
        for tj in trajectories:
            src, src_len, master_mask = self.trajectory2input(tj)
            batch_src.append(src)
            batch_src_len.append(src_len)
            batch_mask.append(master_mask)
        return batch_src, batch_src_len, batch_mask

    def init(
            self, is_training: bool, load_best: bool = False,
            restore_from: Optional[str] = None) -> None:
        self.is_training = is_training
        if self.model is None:
            self.sess, self.model, self.saver = self.create_model(
                self.is_training)
        self.loaded_ckpt_step = self.load_model(
            self.sess, self.model, self.saver, restore_from, load_best)

        if self.is_training:
            if self.loaded_ckpt_step > 0:
                self.create_or_reload_target_model(restore_from)
            train_summary_dir = os.path.join(
                self.model_dir, "summaries", "train")
            self.train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, self.sess.graph)

    def save_best_model(self) -> None:
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
        :param restore_from:
        :return:
        """
        if self.target_sess is None:
            self.debug("create target model ...")
            (self.target_sess, self.target_model, self.target_saver
             ) = self.create_model(is_training=False, device=self.d4target)
        else:
            pass
        trained_step = self.load_model(
            self.target_sess, self.target_model, self.target_saver,
            restore_from, load_best=False)
        self.debug(
            "load target model from trained step {}".format(trained_step))


class BaseAgent(Logging):
    """
    Base agent class that using
     1. action collector
     2. trajectory collector
     3. floor plan collector
     4. tree memory storage and sampling
    """

    def __init__(self, hp: HParams, model_dir: str) -> None:
        super(BaseAgent, self).__init__()
        self.model_dir = model_dir

        self.tjs_prefix = "trajectories"
        self.action_prefix = "actions"
        self.memo_prefix = "memo"
        self.fp_prefix = "floor_plan"
        self.stc_prefix = "state_text"

        self.inv_direction = {
            ACT.gs: ACT.gn, ACT.gn: ACT.gs,
            ACT.ge: ACT.gw, ACT.gw: ACT.ge}

        self.hp, self.tokenizer = self.init_tokens(hp)
        self.info(output_hparams(self.hp))

        core_class = core_name2clazz(self.hp.core_clazz)
        self.core: BaseCore = core_class(
            self.hp, self.model_dir, self.tokenizer)

        self.tjs: Optional[Trajectory[ActionMaster]] = None
        self.memo: Optional[TreeMemory] = None
        self.actor: Optional[ActionCollector] = None
        self.floor_plan: Optional[FloorPlanCollector] = None
        self.stc: Optional[Trajectory[ObsInventory]] = None

        self._initialized: bool = False
        self._episode_has_started: bool = False
        self.total_t: int = 0
        self.in_game_t: int = 0
        # eps decaying test for all-tiers
        # self.eps_getter = ScannerDecayEPS(
        #     decay_step=10000000, decay_range=1000000)
        self.eps_getter = LinearDecayedEPS(
            decay_step=self.hp.annealing_eps_t,
            init_eps=self.hp.init_eps, final_eps=self.hp.final_eps)
        self.eps: float = 0.
        self.is_training: bool = True
        self._stale_tids: List[int] = []
        self._last_action_mask: Optional[np.ndarray] = None
        self._last_sys_action_mask: Optional[np.ndarray] = None
        self._last_action: Optional[ActionDesc] = None
        self._cumulative_score = 0
        self._cumulative_penalty = 0
        self._prev_last_action: Optional[str] = None
        self._prev_master: Optional[str] = None
        self._prev_place: Optional[str] = None
        self._curr_place: Optional[str] = None
        self.game_id: Optional[str] = None
        self._cnt_action: Optional[Dict[int, float]] = None
        self._largest_valid_tag = 0
        self._stale_tags: Optional[List[int]] = None

    @classmethod
    def report_status(cls, lst_of_status: List[Tuple[str, object]]) -> str:
        return ', '.join(
            map(lambda k_v: '{}: {}'.format(k_v[0], k_v[1]), lst_of_status))

    @classmethod
    def get_path_tags(cls, path: str, prefix: str) -> List[int]:
        """
        Get tag from a path of saved objects. E.g. actions-100.npz
        100 will be extracted
        Make sure the item to be extracted is saved with suffix of npz.
        :param path:
        :param prefix:
        :return:
        """
        all_paths = glob.glob(
            os.path.join(path, "{}-*.npz".format(prefix)), recursive=False)
        tags = list(
            map(lambda fn: int(os.path.splitext(fn)[0].split("-")[1]),
                map(lambda p: os.path.basename(p), all_paths)))
        return tags

    @classmethod
    def clip_reward(cls, reward: float) -> float:
        """clip reward into [-1, 1]"""
        return max(min(reward, 1), -1)

    @classmethod
    def get_room_name(cls, master: str) -> Optional[str]:
        """
        Extract and lower room name.
        Return None if not exist.
        :param master:
        :return:
        """
        room_regex = r"^\s*-= (.*) =-.*"
        room_search = re.search(room_regex, master)
        if room_search is not None:
            room_name = room_search.group(1).lower()
        else:
            room_name = None
        return room_name

    @classmethod
    def select_additional_infos(cls) -> EnvInfos:
        """
        additional information needed when playing the game
        requested infos here are required to run the Agent
        """
        return EnvInfos(
            description=True,
            inventory=True,
            max_score=True,
            won=True,
            lost=True,
            admissible_commands=True)

    @classmethod
    def get_bert_tokenizer(cls, hp: HParams) -> Tuple[HParams, Tokenizer]:
        tokenizer = BertTokenizer(
            vocab_file=conventions.bert_vocab_file, do_lower_case=True)
        new_hp = copy_hparams(hp)
        # set vocab info
        new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
        new_hp.set_hparam("padding_val", conventions.bert_padding_token)
        new_hp.set_hparam("unk_val", conventions.bert_unk_token)
        new_hp.set_hparam("cls_val", conventions.bert_cls_token)
        new_hp.set_hparam("sep_val", conventions.bert_sep_token)
        new_hp.set_hparam("mask_val", conventions.bert_mask_token)
        new_hp.set_hparam("sos", conventions.bert_sos_token)
        new_hp.set_hparam("eos", conventions.bert_eos_token)

        # set special token ids
        new_hp.set_hparam(
            'padding_val_id', tokenizer.vocab[conventions.bert_padding_token])
        assert new_hp.padding_val_id == 0, "padding should be indexed as 0"
        new_hp.set_hparam(
            'unk_val_id', tokenizer.vocab[conventions.bert_unk_token])
        # bert specific tokens
        new_hp.set_hparam(
            'cls_val_id', tokenizer.vocab[conventions.bert_cls_token])
        new_hp.set_hparam(
            'sep_val_id', tokenizer.vocab[conventions.bert_sep_token])
        new_hp.set_hparam(
            'mask_val_id', tokenizer.vocab[conventions.bert_mask_token])
        new_hp.set_hparam(
            "sos_id", tokenizer.vocab[conventions.bert_sos_token])
        new_hp.set_hparam(
            "eos_id", tokenizer.vocab[conventions.bert_eos_token])
        return new_hp, tokenizer

    @classmethod
    def get_albert_tokenizer(cls, hp: HParams) -> Tuple[HParams, Tokenizer]:
        tokenizer = AlbertTokenizer(
            vocab_file=conventions.albert_vocab_file,
            do_lower_case=True,
            spm_model_file=conventions.albert_spm_path)
        new_hp = copy_hparams(hp)
        # make sure that padding_val is indexed as 0.
        new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
        new_hp.set_hparam("padding_val", conventions.albert_padding_token)
        new_hp.set_hparam("unk_val", conventions.albert_unk_token)
        new_hp.set_hparam("cls_val", conventions.albert_cls_token)
        new_hp.set_hparam("sep_val", conventions.albert_sep_token)
        new_hp.set_hparam("mask_val", conventions.albert_mask_token)

        new_hp.set_hparam(
            'padding_val_id', tokenizer.vocab[conventions.albert_padding_token])
        assert new_hp.padding_val_id == 0, "padding should be indexed as 0"
        new_hp.set_hparam(
            'unk_val_id', tokenizer.vocab[conventions.albert_unk_token])
        new_hp.set_hparam(
            'cls_val_id', tokenizer.vocab[conventions.albert_cls_token])
        new_hp.set_hparam(
            'sep_val_id', tokenizer.vocab[conventions.albert_sep_token])
        new_hp.set_hparam(
            'mask_val_id', tokenizer.vocab[conventions.albert_mask_token])
        return new_hp, tokenizer

    @classmethod
    def get_nltk_tokenizer(cls, hp: HParams) -> Tuple[HParams, Tokenizer]:
        tokenizer = NLTKTokenizer(
            vocab_file=conventions.nltk_vocab_file, do_lower_case=True)
        new_hp = copy_hparams(hp)
        new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
        new_hp.set_hparam("padding_val", conventions.nltk_padding_token)
        new_hp.set_hparam("unk_val", conventions.nltk_unk_token)
        new_hp.set_hparam("sos", conventions.nltk_sos_token)
        new_hp.set_hparam("eos", conventions.nltk_eos_token)

        new_hp.set_hparam(
            'padding_val_id', tokenizer.vocab[conventions.nltk_padding_token])
        assert new_hp.padding_val_id == 0, "padding should be indexed as 0"
        new_hp.set_hparam(
            'unk_val_id', tokenizer.vocab[conventions.nltk_unk_token])
        new_hp.set_hparam('sos_id', tokenizer.vocab[conventions.nltk_sos_token])
        new_hp.set_hparam('eos_id', tokenizer.vocab[conventions.nltk_eos_token])
        return new_hp, tokenizer

    @classmethod
    def init_tokens(cls, hp: HParams) -> Tuple[HParams, Tokenizer]:
        """
        Note that BERT must use bert vocabulary.
        :param hp:
        :return:
        """
        if hp.tokenizer_type.lower() == "bert":
            new_hp, tokenizer = cls.get_bert_tokenizer(hp)
        elif hp.tokenizer_type.lower() == "albert":
            new_hp, tokenizer = cls.get_albert_tokenizer(hp)
        elif hp.tokenizer_type.lower() == "nltk":
            new_hp, tokenizer = cls.get_nltk_tokenizer(hp)
        else:
            raise ValueError(
                "Unknown tokenizer type: {}".format(hp.tokenizer_type))
        return new_hp, tokenizer

    @classmethod
    def get_admissible_actions(
            cls, infos: Dict[str, List[Any]]) -> List[str]:
        return [a.lower() for a in infos[INFO_KEY.actions][0]]

    @classmethod
    def _compute_game_id(cls, infos: Dict[str, List[Any]]) -> str:
        assert INFO_KEY.desc in infos, "request description is required"
        assert INFO_KEY.inventory in infos, "request inventory is required"
        starter = "{}\n{}".format(
            infos[INFO_KEY.desc][0], infos[INFO_KEY.inventory][0])
        return get_hash(starter)

    def init_actions(
            self, hp: HParams, tokenizer: Tokenizer, action_path: str,
            with_loading=True) -> ActionCollector:
        action_collector = ActionCollector(
            tokenizer=tokenizer,
            n_tokens=hp.n_tokens_per_action,
            unk_val_id=hp.unk_val_id,
            padding_val_id=hp.padding_val_id)
        if with_loading:
            try:
                action_collector.load_actions(action_path)
            except IOError as e:
                self.info("load actions error: \n{}".format(e))
        return action_collector

    def init_trajectory(
            self, hp: HParams, tjs_path: str, with_loading=True) -> Trajectory:
        tjs = Trajectory[ActionMaster](num_turns=hp.num_turns)
        if with_loading:
            try:
                tjs.load_tjs(tjs_path)
            except IOError as e:
                self.info("load trajectory error: \n{}".format(e))
        return tjs

    def init_state_text(self, state_text_path, with_loading=True):
        stc = Trajectory[ObsInventory](num_turns=1)
        if with_loading:
            try:
                stc.load_tjs(state_text_path)
            except IOError as e:
                self.info("load trajectory error: \n{}".format(e))
        return stc

    def init_memo(
            self, hp: HParams, memo_path: str, with_loading=True) -> TreeMemory:
        memory = TreeMemory(capacity=hp.replay_mem)
        if with_loading:
            try:
                memory.load_memo(memo_path)
            except IOError as e:
                self.info("load memory error: \n{}".format(e))
        return memory

    def init_floor_plan(
            self, fp_path: str, with_loading=True) -> FloorPlanCollector:
        fp = FloorPlanCollector()
        if with_loading:
            try:
                fp.load_fps(fp_path)
            except IOError as e:
                self.info("load floor plan error: \n{}".format(e))
        return fp

    def get_a_random_action(self, action_mask: np.ndarray) -> ActionDesc:
        """
        Select a random action according to action mask
        :param action_mask:
        :return:
        """
        action_idx = np.random.choice(action_mask)
        action_desc = ActionDesc(
            action_type=ACT_TYPE.rnd,
            action_idx=action_idx,
            token_idx=self.actor.action_matrix[action_idx],
            action_len=self.actor.action_len[action_idx],
            action=self.actor.actions[action_idx],
            q_actions=None)
        return action_desc

    def train(self) -> None:
        """
        call train() before performing training
        :return:
        """
        self.is_training = True
        self._init()

    def eval(self, load_best=True) -> None:
        """
        call eval() before performing evaluation
        :param load_best: load from best weights, or from last weights
        :return:
        """
        self.is_training = False
        self._init(load_best)

    def reset(self, restore_from: Optional[str] = None) -> None:
        """
        reset is only used for evaluation during training
        do not use it at anywhere else.
        """
        self.is_training = False
        self._initialized = False
        self._init(load_best=False, restore_from=restore_from)

    def _init(
            self, load_best=False, restore_from: Optional[str] = None) -> None:
        """
        load actions, trajectories, memory, model, etc.
        """
        if self._initialized:
            self.error("the agent was initialized")
            return
        self._init_impl(load_best, restore_from)
        self._initialized = True

    def _get_context_obj_path_w_tag(self, prefix: str, tag: int) -> str:
        return pjoin(
            self.model_dir, "{}-{}.npz".format(prefix, tag))

    def _get_context_obj_path(self, prefix: str) -> str:
        return self._get_context_obj_path_w_tag(prefix, self._largest_valid_tag)

    def _get_context_obj_new_path(self, prefix: str) -> str:
        return self._get_context_obj_path_w_tag(prefix, self.total_t)

    def _load_context_objs(self) -> None:
        valid_tags = self.get_compatible_snapshot_tag()
        self._largest_valid_tag = max(valid_tags) if len(valid_tags) != 0 else 0
        self.info("try to load from tag: {}".format(self._largest_valid_tag))

        action_path = self._get_context_obj_path(self.action_prefix)
        tjs_path = self._get_context_obj_path(self.tjs_prefix)
        memo_path = self._get_context_obj_path(self.memo_prefix)
        fp_path = self._get_context_obj_path(self.fp_prefix)
        stc_path = self._get_context_obj_path(self.stc_prefix)

        # always loading actions to avoid different action index for DQN
        self.actor = self.init_actions(
            self.hp, self.tokenizer, action_path,
            with_loading=self.is_training)
        self.tjs = self.init_trajectory(
            self.hp, tjs_path, with_loading=self.is_training)
        self.memo = self.init_memo(
            self.hp, memo_path, with_loading=self.is_training)
        self.floor_plan = self.init_floor_plan(
            fp_path, with_loading=self.is_training)
        # load stc
        self.stc = self.init_state_text(stc_path, with_loading=True)

    def _init_impl(
            self, load_best=False, restore_from: Optional[str] = None) -> None:
        self._load_context_objs()
        self.core.init(
            is_training=self.is_training, load_best=load_best,
            restore_from=restore_from)

        if self.is_training:
            if self.hp.start_t_ignore_model_t:
                self.total_t = min(
                    self.hp.observation_t,
                    len(self.memo) if self.memo is not None else 0)
            else:
                self.total_t = min(
                    self.hp.observation_t + self.core.loaded_ckpt_step,
                    len(self.memo) if self.memo is not None else 0)
        else:
            self.eps = 0
            self.total_t = 0

    def _start_episode(
            self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        """
        Prepare the agent for the upcoming episode.
        :param obs: initial feedback from each game
        :param infos: additional infors of each game
        :return:
        """
        if not self._initialized:
            self._init()
        self._start_episode_impl(obs, infos)

    def _start_episode_impl(
            self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        self.tjs.add_new_tj()
        self.stc.add_new_tj(tid=self.tjs.get_current_tid())
        self.game_id = self._compute_game_id(infos)
        self.info("game id: {}".format(self.game_id))
        self.actor.add_new_episode(gid=self.game_id)
        self.floor_plan.add_new_episode(eid=self.game_id)
        self.in_game_t = 0
        self._cumulative_score = 0
        self._episode_has_started = True
        self._prev_place = None
        self._curr_place = None
        self._cnt_action = dict()

    def _end_episode(
            self, obs: List[str], scores: List[int],
            infos: Dict[str, List[Any]]) -> None:
        """
        tell the agent the episode has terminated
        :param obs: previous command's feedback for each game
        :param scores: score obtained so far for each game
        :param infos: additional infos of each game
        :return:
        """
        self.info(self.report_status(
            [("training", self.is_training),
             ("#steps", self.in_game_t),
             ("score", scores[0]),
             ("won", infos[INFO_KEY.won][0]),
             ("lost", infos[INFO_KEY.lost][0]),
             ("last eps", self.eps)]))
        self._episode_has_started = False
        self._last_action_mask = None
        self._last_sys_action_mask = None
        self.game_id = None
        self._last_action = None
        self._cumulative_penalty = 0
        self._prev_last_action = None
        self._prev_master = None

    def _delete_stale_context_objs(self) -> None:
        valid_tags = self.get_compatible_snapshot_tag()
        if len(valid_tags) > self.hp.max_snapshot_to_keep:
            self._stale_tags = list(reversed(sorted(
                valid_tags)))[self.hp.max_snapshot_to_keep:]
            self.info("tags to be deleted: {}".format(self._stale_tags))
            for tag in self._stale_tags:
                prm(self._get_context_obj_path_w_tag(self.memo_prefix, tag))
                prm(self._get_context_obj_path_w_tag(self.tjs_prefix, tag))
                prm(self._get_context_obj_path_w_tag(self.action_prefix, tag))
                prm(self._get_context_obj_path_w_tag(self.fp_prefix, tag))
                prm(self._get_context_obj_path_w_tag(self.stc_prefix, tag))

    def _save_context_objs(self) -> None:
        action_path = self._get_context_obj_new_path(self.action_prefix)
        tjs_path = self._get_context_obj_new_path(self.tjs_prefix)
        memo_path = self._get_context_obj_new_path(self.memo_prefix)
        fp_path = self._get_context_obj_new_path(self.fp_prefix)
        stc_path = self._get_context_obj_new_path(self.stc_prefix)

        self.memo.save_memo(memo_path)
        self.tjs.save_tjs(tjs_path)
        self.actor.save_actions(action_path)
        self.floor_plan.save_fps(fp_path)
        self.stc.save_tjs(stc_path)

    def save_snapshot(self) -> None:
        self.info('save snapshot of the agent')
        self._save_context_objs()
        self._delete_stale_context_objs()
        self._clean_stale_context(self._stale_tids)
        # notice that we should not save hparams when evaluating
        # that's why I move this function calling here from __init__
        save_hparams(self.hp, pjoin(self.model_dir, 'hparams.json'))

    def get_compatible_snapshot_tag(self) -> List[int]:
        action_tags = self.get_path_tags(self.model_dir, self.action_prefix)
        memo_tags = self.get_path_tags(self.model_dir, self.memo_prefix)
        tjs_tags = self.get_path_tags(self.model_dir, self.tjs_prefix)
        fp_tags = self.get_path_tags(self.model_dir, self.fp_prefix)
        stc_tags = self.get_path_tags(self.model_dir, self.stc_prefix)

        valid_tags = set(action_tags)
        valid_tags.intersection_update(memo_tags)
        valid_tags.intersection_update(tjs_tags)
        valid_tags.intersection_update(fp_tags)
        valid_tags.intersection_update(stc_tags)

        return list(valid_tags)

    def is_time_to_save(self) -> bool:
        trained_steps = self.total_t - self.hp.observation_t + 1
        return (trained_steps % self.hp.save_gap_t == 0) and (trained_steps > 0)

    def go_with_floor_plan(self, actions: List[str]) -> List[str]:
        """
        Update go-cardinal actions into go-room actions, if floor plan exists
        :param actions:
        :return:
        """
        local_map = self.floor_plan.get_map(self._curr_place)
        return (
            ["{} to {}".format(a, local_map.get(a))
             if a in local_map else a for a in actions])

    def random_walk_for_collecting_fp(
            self, actions: List[str], all_actions: List[str]) -> ActionDesc:
        action_idx, action = None, None

        if self.hp.collect_floor_plan:
            # collecting floor plan by choosing random action
            # if there are still go actions without room name
            # Notice that only choosing "go" actions cannot finish
            # collecting floor plan because there is the need to open doors
            # Notice also that only allow random walk in the first 50 steps
            cardinal_go = list(filter(
                lambda a: a.startswith("go") and len(a.split()) == 2, actions))
            if self.in_game_t < 50 and len(cardinal_go) != 0:
                open_actions = list(
                    filter(lambda a: a.startswith("open"), actions))
                admissible_actions = cardinal_go + open_actions
                self.debug("admissible actions for random walk: {}".format(
                    admissible_actions))
                action = np.random.choice(admissible_actions)
                action_idx = all_actions.index(action)
            else:
                pass
        else:
            pass
        action_desc = ActionDesc(
            action_type=ACT_TYPE.rnd_walk, action_idx=action_idx,
            token_idx=self.actor.action_matrix[action_idx],
            action_len=self.actor.action_len[action_idx],
            action=action, q_actions=None)
        return action_desc

    def get_policy_action(self, action_mask: np.ndarray) -> ActionDesc:
        trajectory = self.tjs.fetch_last_state()
        state = self.stc.fetch_last_state()[0]
        policy_action_desc = self.core.get_a_policy_action(
            trajectory, state, self.actor.action_matrix,
            self.actor.action_len, self.actor.actions, action_mask,
            self._cnt_action)
        return policy_action_desc

    def choose_action(
            self, actions: List[str], all_actions: List[str],
            action_mask: np.ndarray, instant_reward: float) -> ActionDesc:
        # when q_actions is required to get, this should be True
        if self.hp.compute_policy_action_every_step:
            policy_action_desc = self.get_policy_action(action_mask)
        else:
            policy_action_desc = None

        action_desc = self.random_walk_for_collecting_fp(actions, all_actions)
        if action_desc.action_idx is None:
            if random.random() < self.eps:
                action_desc = self.get_a_random_action(action_mask)
            else:
                if policy_action_desc:
                    action_desc = policy_action_desc
                else:
                    action_desc = self.get_policy_action(action_mask)

        final_action_desc = ActionDesc(
            action_type=action_desc.action_type,
            action_idx=action_desc.action_idx,
            action_len=action_desc.action_len,
            token_idx=action_desc.token_idx,
            action=action_desc.action,
            q_actions=(
                policy_action_desc.q_actions if policy_action_desc else None))
        return final_action_desc

    def get_raw_instant_reward(self, score: float) -> float:
        """raw instant reward between two consecutive scores"""
        instant_reward = score - self._cumulative_score
        self._cumulative_score = score
        return instant_reward

    def get_repetition_penalty(
            self, master: str, instant_reward: float) -> float:
        """
        add a penalty of self._cumulative_penalty if the current Action-Master
        repeats the failure of last Action-Master.
        """
        if (master == self._prev_master and self._last_action is not None
                and self._last_action.action == self._prev_last_action and
                instant_reward < 0):
            self._cumulative_penalty -= 0.1
        else:
            self._prev_last_action = (
                self._last_action.action
                if self._last_action is not None else None)
            self._prev_master = master
            self._cumulative_penalty = 0.
        return self._cumulative_penalty

    def get_instant_reward(
            self, score: float, master: str, is_terminal: bool,
            won: bool, lost: bool) -> float:
        # there are three scenarios of game termination
        # 1. you won      --> encourage this action
        # 2. you lost     --> discourage this action
        # 3. out of step  --> do nothing
        raw_instant_reward = self.get_raw_instant_reward(score)
        instant_reward = raw_instant_reward
        if is_terminal:
            if won:
                instant_reward = max(1., raw_instant_reward)
            elif lost:
                instant_reward = min(-1., raw_instant_reward)
            else:
                pass
        # add repetition penalty and per-step penalty
        if self.hp.use_step_wise_reward:
            instant_reward += (
                self.get_repetition_penalty(master, instant_reward) + (-0.1))
        instant_reward = self.clip_reward(instant_reward)
        return instant_reward

    def collect_floor_plan(self, master: str, prev_place: str) -> str:
        """
        collect floor plan with latest master.
        if the current place doesn't match the previous place, and a go action
        is used to get the master, then we need to update the floor plan.

        :param master:
        :param prev_place: the name of previous place
        :return: the name of current place
        """
        room_name = self.get_room_name(master)
        curr_place = room_name if room_name is not None else prev_place

        if (curr_place != prev_place and
                self._last_action is not None and
                self._last_action.action in self.inv_direction):
            self.floor_plan.extend(
                [(prev_place, self._last_action.action, curr_place),
                 (curr_place, self.inv_direction[self._last_action.action],
                  prev_place)])
        return curr_place

    def _prepare_other_train_data(self, b_memory: List[Memolet]) -> Any:
        return None

    def train_one_batch(self) -> None:
        """
        Train one batch of samples.
        Load target model if not exist, save current model when necessary.
        """
        # prepare other data
        # put it as the first, in case of using multi-threading
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)
        other_train_data = self._prepare_other_train_data(b_memory)

        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        action_id = [m.aid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        reward = [m.reward for m in b_memory]
        is_terminal = [m.is_terminal for m in b_memory]
        pre_action_mask = [m.action_mask for m in b_memory]
        post_action_mask = [m.next_action_mask for m in b_memory]

        post_trajectories = self.tjs.fetch_batch_states(trajectory_id, state_id)
        pre_trajectories = self.tjs.fetch_batch_pre_states(
            trajectory_id, state_id)

        post_states = [
            state[0] for state in
            self.stc.fetch_batch_states(trajectory_id, state_id)]
        pre_states = [
            state[0] for state in
            self.stc.fetch_batch_pre_states(trajectory_id, state_id)]

        action_len = (
            [self.actor.get_action_len(gid) for gid in game_id])
        action_matrix = (
            [self.actor.get_action_matrix(gid) for gid in game_id])

        b_weight = self.core.train_one_batch(
            pre_trajectories=pre_trajectories,
            post_trajectories=post_trajectories,
            pre_states=pre_states,
            post_states=post_states,
            action_matrix=action_matrix,
            action_len=action_len,
            pre_action_mask=pre_action_mask,
            post_action_mask=post_action_mask,
            dones=is_terminal,
            rewards=reward,
            action_idx=action_id,
            b_weight=b_weight,
            step=self.total_t,
            others=other_train_data)

        self.memo.batch_update(b_idx, b_weight)

        if self.is_time_to_save():
            self.save_snapshot()
            self.core.save_model()
            self.core.create_or_reload_target_model()

    def _clean_stale_context(self, tids: List[int]) -> None:
        self.debug("tjs deletes {}".format(tids))
        self.tjs.request_delete_keys(tids)
        self.stc.request_delete_keys(tids)

    def update_status(
            self, obs: List[str], scores: List[float], dones: List[bool],
            infos: Dict[str, List[Any]]) -> Tuple[str, float]:
        self._prev_place = self._curr_place
        master = infos[INFO_KEY.desc][0] if self.in_game_t == 0 else obs[0]
        instant_reward = self.get_instant_reward(
            scores[0], obs[0], dones[0],
            infos[INFO_KEY.won][0], infos[INFO_KEY.lost][0])

        self.debug("master: {}, raw reward: {}, instant reward: {}".format(
            master, scores[0], instant_reward))
        if self.hp.collect_floor_plan:
            self._curr_place = self.collect_floor_plan(master, self._prev_place)
        else:
            self._curr_place = None

        return master, instant_reward

    def collect_new_sample(
            self, master: str, instant_reward: float, dones: List[bool],
            infos: Dict[str, List[Any]]) -> Tuple[
            List[str], List[str], np.ndarray, np.ndarray, float]:

        self.tjs.append(ActionMaster(
            action=self._last_action.action if self._last_action else "",
            master=master))

        if not dones[0]:
            state = ObsInventory(
                obs=infos[INFO_KEY.desc][0],
                inventory=infos[INFO_KEY.inventory][0])
        else:
            obs = (
                "terminal and win" if infos[INFO_KEY.won]
                else "terminal and lose")
            state = ObsInventory(obs=obs, inventory="")
        self.stc.append(state)

        admissible_actions = self.get_admissible_actions(infos)
        sys_action_mask = self.actor.extend(admissible_actions)
        effective_actions = self.go_with_floor_plan(admissible_actions)
        action_mask = self.actor.extend(effective_actions)
        all_actions = self.actor.actions
        # TODO: use all actions instead of using admissible actions
        # action_mask = self.actor.extend(all_actions)
        self.debug("effective actions: {}".format(effective_actions))

        if self.tjs.get_last_sid() > 0:
            memo_let = Memolet(
                tid=self.tjs.get_current_tid(),
                sid=self.tjs.get_last_sid(),
                gid=self.game_id,
                aid=self._last_action.action_idx,
                token_id=self._last_action.token_idx,
                a_len=self._last_action.action_len,
                a_type=self._last_action.action_type,
                reward=instant_reward,
                is_terminal=dones[0],
                action_mask=self._last_action_mask,
                sys_action_mask=self._last_sys_action_mask,
                next_action_mask=action_mask,
                next_sys_action_mask=sys_action_mask,
                q_actions=self._last_action.q_actions
            )
            self.debug("memo_let: {}".format(memo_let))
            original_data = self.memo.append(memo_let)
            if isinstance(original_data, Memolet):
                if original_data.is_terminal:
                    self._stale_tids.append(original_data.tid)

        return (effective_actions, all_actions, action_mask, sys_action_mask,
                instant_reward)

    def next_step_action(
            self, actions: List[str], all_actions: List[str],
            instant_reward: float,
            action_mask: np.ndarray, sys_action_mask: np.ndarray) -> str:

        if self.is_training:
            self.eps = self.eps_getter.eps(self.total_t - self.hp.observation_t)
        else:
            pass
        self._last_action = self.choose_action(
            actions, all_actions, action_mask, instant_reward)
        action = self._last_action.action
        action_idx = self._last_action.action_idx

        self.debug("action: {}".format(self._last_action))

        if self._last_action.action_type == ACT_TYPE.policy_drrn:
            if action_idx not in self._cnt_action:
                self._cnt_action[action_idx] = 0.
            self._cnt_action[action_idx] += 0.1

        self._last_action_mask = action_mask
        self._last_sys_action_mask = sys_action_mask
        # revert back go actions for the game playing
        if action.startswith("go"):
            action = " ".join(action.split()[:2])
        return action

    def act(self, obs: List[str], scores: List[int], dones: List[bool],
            infos: Dict[str, List[Any]]) -> Optional[List[str]]:
        """
        Acts upon the current list of observations.
        One text command must be returned for each observation.
        :param obs:
        :param scores: score obtained so far for each game
        :param dones: whether a game is finished
        :param infos:
        :return: if all dones, return None, else return actions

        Notes:
            Commands returned for games marked as `done` have no effect.
            The states for finished games are simply copy over until all
            games are done.
        """
        if not self._episode_has_started:
            self._start_episode(obs, infos)

        assert len(obs) == 1, "cannot handle batch game training"
        master, instant_reward = self.update_status(obs, scores, dones, infos)
        (actions, all_actions, action_mask, sys_action_mask, instant_reward
         ) = self.collect_new_sample(master, instant_reward, dones, infos)
        # notice the position of all(dones)
        # make sure add the last action-master pair into memory
        if all(dones):
            self._end_episode(obs, scores, infos)
            return None

        player_t = self.next_step_action(
            actions, all_actions, instant_reward, action_mask, sys_action_mask)
        if self.is_training and self.total_t >= self.hp.observation_t:
            self.train_one_batch()
        self.total_t += 1
        self.in_game_t += 1
        return [player_t] * len(obs)
