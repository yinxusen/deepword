import glob
import os
import random
import re
from abc import ABC
from os import remove as prm
from os.path import join as pjoin
from typing import Any

import tensorflow as tf
import tensorflow.contrib.training.HParams as HParams
from bitarray import bitarray
from tensorflow import Session
from tensorflow.python.client import device_lib
from tensorflow.summary import FileWriter
from tensorflow.train import Saver
from textworld import EnvInfos

from deeptextworld.action import ActionCollector
from deeptextworld.agents.utils import *
from deeptextworld.dependency_parser import DependencyParserReorder
from deeptextworld.floor_plan import FloorPlanCollector
from deeptextworld.hparams import save_hparams, output_hparams, copy_hparams
from deeptextworld.models.dqn_model import DQNModel
from deeptextworld.trajectory import Trajectory
from deeptextworld.tree_memory import TreeMemory
from deeptextworld.utils import ctime
from deeptextworld.utils import model_name2clazz, get_hash


class BaseCore(Logging, ABC):
    def get_a_policy_action(
            self,
            trajectory: List[ActionMaster],
            state: Optional[ObsInventory],
            action_matrix: np.ndarray,
            action_len: np.ndarray, actions: List[str],
            action_mask: np.ndarray) -> ActionDesc:
        raise NotImplementedError()

    def save_model(self) -> None:
        raise NotImplementedError()

    def init(self, load_best=False, restore_from: Optional[str] = None) -> None:
        raise NotImplementedError()

    def train_one_batch(
            self,
            pre_trajectories: List[List[ActionMaster]],
            post_trajectories: List[List[ActionMaster]],
            pre_states: Optional[List[ObsInventory]],
            post_states: Optional[List[ObsInventory]],
            action_matrix: List[np.ndarray],
            action_len: List[np.ndarray],
            pre_action_mask: np.ndarray,
            post_action_mask: np.ndarray,
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
        super(TFCore, self).__init__()
        self.hp: HParams = hp
        self.model_dir: str = model_dir
        self.model: Optional[DQNModel] = None
        self.target_model: Optional[DQNModel] = None
        self.loaded_ckpt_step: int = 0
        self.sess: Optional[Session] = None
        self.target_sess: Optional[Session] = None
        self.is_training: bool = True
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
        devices = [d.name for d in device_lib.list_local_devices()
                   if d.device_type == "GPU"]
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

    def save_model(self) -> None:
        self.info('save model')
        self.saver.save(
            self.sess, self.ckpt_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=self.model.graph))

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
        return model_creator.get_train_model(self.hp, device)

    def create_eval_model_instance(self, device):
        model_creator = model_name2clazz(self.hp.model_creator)
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

    def init(
            self, load_best=False, restore_from: Optional[str] = None) -> None:
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

        self.core: Optional[BaseCore] = None

        self.tjs: Optional[Trajectory[ActionMaster]] = None
        self.memo: Optional[TreeMemory] = None
        self.actor: Optional[ActionCollector] = None
        self.floor_plan: Optional[FloorPlanCollector] = None
        self.dp: Optional[DependencyParserReorder] = None
        self.stc: Optional[Trajectory[ObsInventory]] = None

        self._initialized: bool = False
        self._episode_has_started: bool = False
        self.total_t: int = 0
        self.in_game_t: int = 0
        # eps decaying test for all-tiers
        self.eps_getter = ScannerDecayEPS(
            decay_step=10000000, decay_range=1000000)
        # self.eps_getter = LinearDecayedEPS(
        #     decay_step=self.hp.annealing_eps_t,
        #     init_eps=self.hp.init_eps, final_eps=self.hp.final_eps)
        self.eps: float = 0.
        self.is_training: bool = True
        self.snapshot_saved = False
        self.epoch_start_t = 0

        self._stale_tids: List[int] = []

        self._last_actions_mask: Optional[bytes] = None
        self._last_action: Optional[ActionDesc] = None

        self._cumulative_score = 0
        self._cumulative_penalty = -0.1
        self._prev_last_action: Optional[str] = None
        self._prev_master: Optional[str] = None
        self._prev_place: Optional[str] = None
        self._curr_place: Optional[str] = None

        self.game_id: Optional[str] = None
        self._theme_words: Dict[str, List[str]] = {}

        self._see_cookbook = False
        self._cnt_action: Optional[np.ndarray] = None

        self._largest_valid_tag = 0
        self._stale_tags: Optional[List[int]] = None

    @classmethod
    def report_status(cls, lst_of_status: List[Tuple[str, object]]) -> str:
        return ', '.join(
            map(lambda k_v: '{}: {}'.format(k_v[0], k_v[1]), lst_of_status))

    @classmethod
    def from_bytes(cls, byte_action_masks: List[bytes]) -> np.ndarray:
        """
        Convert a list of byte-array masks to a list of np-array masks.
        :param byte_action_masks:
        :return:
        """
        vec_action_masks = []
        for mask in byte_action_masks:
            bit_mask = bitarray(endian='little')
            bit_mask.frombytes(mask)
            bit_mask[-1] = False
            vec_action_masks.append(bit_mask.tolist())
        return np.asarray(vec_action_masks, dtype=np.int32)

    @classmethod
    def get_theme_words(cls, recipe: str) -> List[str]:
        theme_regex = r".*Ingredients:<\|>(.*)<\|>Directions.*"
        theme_words_search = re.search(
            theme_regex, recipe.replace("\n", "<|>"))
        if theme_words_search:
            theme_words = theme_words_search.group(1)
            theme_words = list(
                filter(lambda w: w != "",
                       map(lambda w: w.strip(), theme_words.split("<|>"))))
        else:
            theme_words = None
        return theme_words

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
    def contain_words(cls, sentence: str, words: List[str]) -> bool:
        """
        Does sentence contain any word in words?
        :param sentence:
        :param words:
        :return:
        """
        return any(map(lambda w: w in sentence, words))

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
    def negative_response_reward(cls, master: str) -> float:
        """
        To check if the responded master say something negative or not
        :param master:
        :return:
        """
        return 0

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
            admissible_commands=True,
            extras=['recipe'])

    @classmethod
    def init_tokens(cls, hp: HParams) -> Tuple[HParams, Tokenizer]:
        """
        Note that BERT must use bert vocabulary.
        :param hp:
        :return:
        """
        if hp.tokenizer_type == "BERT":
            tokenizer = BertTokenizer(
                vocab_file=hp.vocab_file, do_lower_case=True)
            new_hp = copy_hparams(hp)
            # make sure that padding_val is indexed as 0.
            new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
            new_hp.set_hparam('padding_val_id', tokenizer.vocab[hp.padding_val])
            new_hp.set_hparam('unk_val_id', tokenizer.vocab[hp.unk_val])
            # bert specific tokens
            new_hp.set_hparam('cls_val_id', tokenizer.vocab[hp.cls_val])
            new_hp.set_hparam('sep_val_id', tokenizer.vocab[hp.sep_val])
            new_hp.set_hparam('mask_val_id', tokenizer.vocab[hp.mask_val])
        elif hp.tokenizer_type == "Albert":
            tokenizer = AlbertTokenizer(
                vocab_file=hp.vocab_file, do_lower_case=True,
                spm_model_file=hp.spm_model_file)
            new_hp = copy_hparams(hp)
            # make sure that padding_val is indexed as 0.
            new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
            new_hp.set_hparam('padding_val_id', tokenizer.vocab[hp.padding_val])
            new_hp.set_hparam('unk_val_id', tokenizer.vocab[hp.unk_val])
            # bert specific tokens
            new_hp.set_hparam('cls_val_id', tokenizer.vocab[hp.cls_val])
            new_hp.set_hparam('sep_val_id', tokenizer.vocab[hp.sep_val])
            new_hp.set_hparam('mask_val_id', tokenizer.vocab[hp.mask_val])
        elif hp.tokenizer_type == "NLTK":
            tokenizer = NLTKTokenizer(
                vocab_file=hp.vocab_file, do_lower_case=True)
            new_hp = copy_hparams(hp)
            # make sure that padding_val is indexed as 0.
            new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
            new_hp.set_hparam('padding_val_id', tokenizer.vocab[hp.padding_val])
            new_hp.set_hparam('unk_val_id', tokenizer.vocab[hp.unk_val])
            new_hp.set_hparam('sos_id', tokenizer.vocab[hp.sos])
            new_hp.set_hparam('eos_id', tokenizer.vocab[hp.eos])
        else:
            raise ValueError(
                "Unknown tokenizer type: {}".format(hp.tokenizer_type))
        return new_hp, tokenizer

    def init_actions(
            self, hp: HParams, tokenizer: Tokenizer, action_path: str,
            with_loading=True) -> ActionCollector:
        action_collector = ActionCollector(
            tokenizer,
            hp.n_actions, hp.n_tokens_per_action,
            hp.unk_val_id, hp.padding_val_id, hp.eos_id, hp.pad_eos)
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

    def get_cleaned_master(self, master: str) -> str:
        """
        Cleaned_master is a master that are lowered, tokenized, and then
        concatenated by space.
        :param master:
        :return:
        """
        return " ".join(self.tokenizer.tokenize(master))

    def get_a_random_action(self, action_mask: bytes) -> ActionDesc:
        """
        Select a random action according to action mask
        :param action_mask:
        :return:
        """
        action_mask = self.from_bytes([action_mask])[0]
        mask_idx = np.where(action_mask == 1)[0]
        action_idx = np.random.choice(mask_idx)
        action_desc = ActionDesc(
            action_type=ACT_TYPE.rnd,
            action_idx=action_idx,
            token_idx=self.actor.action_matrix[action_idx],
            action_len=self.actor.action_len[action_idx],
            action=self.actor.actions[action_idx])
        return action_desc

    def get_an_eps_action(self, action_mask: bytes) -> ActionDesc:
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param action_mask:
        """
        if random.random() < self.eps:
            action_desc = self.get_a_random_action(action_mask)
        else:
            action_mask = self.from_bytes([action_mask])[0]
            trajectory = self.tjs.fetch_last_state()
            state = self.stc.fetch_last_state()[0]
            action_desc = self.core.get_a_policy_action(
                trajectory, state, self.actor.action_matrix,
                self.actor.action_len, self.actor.actions, action_mask)
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
        self.core.init(load_best, restore_from)

        if self.is_training:
            if self.hp.start_t_ignore_model_t:
                self.total_t = min(
                    self.hp.observation_t,
                    len(self.memo) if self.memo is not None else 0)
            else:
                self.total_t = (
                        self.core.loaded_ckpt_step + self.hp.observation_t)
        else:
            self.eps = 0
            self.total_t = 0

    def _get_master_starter(
            self, obs: List[str], infos: Dict[str, List[Any]]) -> str:
        assert INFO_KEY.desc in infos, "request description is required"
        assert INFO_KEY.inventory in infos, "request inventory is required"
        return "{}\n{}".format(
            infos[INFO_KEY.desc][0], infos[INFO_KEY.inventory][0])

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
        master_starter = self._get_master_starter(obs, infos)
        self.game_id = get_hash(master_starter)
        self.actor.add_new_episode(eid=self.game_id)
        self.floor_plan.add_new_episode(eid=self.game_id)
        self.in_game_t = 0
        self._cumulative_score = 0
        self._episode_has_started = True
        self._prev_place = None
        self._curr_place = None
        self._cnt_action = np.zeros(self.hp.n_actions)
        if self.game_id not in self._theme_words:
            self._theme_words[self.game_id] = []
        self._per_game_recorder = []
        self._see_cookbook = False
        self.debug("infos: {}".format(infos))

    def mode(self) -> str:
        return "train" if self.is_training else "eval"

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
        # TODO: clear old memory
        # if len(self.memo) > 2 * self.hp.replay_mem:
        #     to_delete_tj_id = self.memo.clear_old_memory()
        #     self.tjs.request_delete_key(to_delete_tj_id)
        self.info(
            "mode: {}, obs: {}, #step: {}, score: {}, won: {},"
            " last_eps: {}".format(
                self.mode(), obs[0], self.in_game_t, scores[0],
                infos[INFO_KEY.won], self.eps))
        self._episode_has_started = False
        self._last_actions_mask = None
        self.game_id = None
        self._last_action = None
        self._cumulative_penalty = -0.1
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
        save_hparams(
            self.hp, pjoin(self.model_dir, 'hparams.json'),
            use_relative_path=True)

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

    def contain_theme_words(
            self, actions: List[str]) -> Tuple[List[str], List[str]]:
        """
        Split actions into two sets - one with theme words; one without them.
        If theme words is empty, all actions are saved in contained, while
        others is empty.
        :param actions:
        :return: contained, others
        """
        if not self._theme_words[self.game_id]:
            self.debug("no theme word found, use all actions")
            return actions, []
        contained = []
        others = []
        for a in actions:
            if self.contain_words(a, self._theme_words[self.game_id]):
                contained.append(a)
            else:
                others.append(a)

        return contained, others

    def filter_admissible_actions(
            self, admissible_actions: List[str]) -> List[str]:
        """
        Filter unnecessary actions.
        :param admissible_actions: raw action given by the game.
        :return:
        """
        contained, others = self.contain_theme_words(admissible_actions)
        actions = [ACT.inventory, ACT.look] + contained
        actions = list(filter(lambda c: not c.startswith("examine"), actions))
        actions = list(filter(lambda c: not c.startswith("close"), actions))
        actions = list(filter(lambda c: not c.startswith("insert"), actions))
        actions = list(filter(lambda c: not c.startswith("eat"), actions))
        # don't drop useful ingredients if not in kitchen
        # while other items can be dropped anywhere
        if self._curr_place != "kitchen":
            actions = list(filter(lambda c: not c.startswith("drop"), actions))
        actions = list(filter(lambda c: not c.startswith("put"), actions))
        other_valid_commands = {
            ACT.prepare_meal, ACT.eat_meal, ACT.examine_cookbook
        }
        actions += list(filter(
            lambda a: a in other_valid_commands, admissible_actions))
        actions += list(filter(
            lambda a: a.startswith("go"), admissible_actions))
        actions += list(filter(
            lambda a: (a.startswith("drop") and
                       all(map(lambda t: t not in a,
                               self._theme_words[self.game_id]))),
            others))
        # meal should never be dropped
        try:
            actions.remove("drop meal")
        except ValueError as _:
            pass
        actions += list(filter(
            lambda a: a.startswith("take") and "knife" in a, others))
        actions += list(filter(lambda a: a.startswith("open"), others))
        # self.debug("previous admissible actions: {}".format(
        #     ", ".join(sorted(admissible_actions))))
        # self.debug("new admissible actions: {}".format(
        #     ", ".join(sorted(actions))))
        actions = list(set(actions))
        return actions

    def go_with_floor_plan(self, actions: List[str]) -> List[str]:
        """
        Update go-cardinal actions into go-room actions, if floor plan exists
        :param actions:
        :return:
        """
        local_map = self.floor_plan.get_map(self._curr_place)
        return (["{} to {}".format(a, local_map.get(a))
                 if a in local_map else a for a in actions])

    def rule_based_policy(
            self, actions: List[str], all_actions: List[str],
            instant_reward: float) -> ActionDesc:
        if (self._last_action is not None and
                self._last_action.action == ACT.prepare_meal and
                instant_reward > 0):
            action = ACT.eat_meal
        elif ACT.examine_cookbook in actions and not self._see_cookbook:
            action = ACT.examine_cookbook
            self._see_cookbook = True
        elif (self._last_action is not None and
              self._last_action.action == ACT.examine_cookbook and
              instant_reward <= 0):
            action = ACT.inventory
        elif (self._last_action is not None and
              self._last_action.action.startswith("take") and
              instant_reward <= 0):
            action = ACT.inventory
        else:
            action = None

        if action is not None:
            if action not in all_actions:
                self.debug("eat meal not in action list, adding it in ...")
                self.actor.extend([action])
                all_actions = self.actor.actions
            action_idx = all_actions.index(action)
        else:
            action_idx = None
        action_desc = ActionDesc(
            action_type=ACT_TYPE.rule, action_idx=action_idx,
            token_idx=self.actor.action_matrix[action_idx],
            action_len=self.actor.action_len[action_idx],
            action=action)
        return action_desc

    def _is_jitter_go(
            self, action_desc: ActionDesc,
            admissible_go_actions: List[str]) -> bool:
        if not (self.hp.jitter_go and
                action_desc.action in admissible_go_actions and
                action_desc.action_type == ACT_TYPE.policy_drrn):
            return False
        else:
            if self.is_training:
                return np.random.random() > 1. - self.hp.jitter_train_prob
            else:
                return np.random.random() > 1. - self.hp.jitter_eval_prob

    def jitter_go_action(
            self, prev_action_desc: ActionDesc, actions: List[str],
            all_actions: List[str]) -> ActionDesc:
        action_desc = None
        admissible_go_actions = list(
            filter(lambda a: a.startswith("go"), actions))
        if self._is_jitter_go(prev_action_desc, admissible_go_actions):
            jitter_go_action = np.random.choice(admissible_go_actions)
            action_idx = all_actions.index(jitter_go_action)
            action = jitter_go_action
            action_desc = ActionDesc(
                action_type=ACT_TYPE.jitter, action_idx=action_idx,
                token_idx=self.actor.action_matrix[action_idx],
                action_len=self.actor.action_len[action_idx],
                action=action)
        else:
            pass
        return action_desc if action_desc is not None else prev_action_desc

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
                self.debug("admissible actions for random walk:\n{}".format(
                    admissible_actions))
                _, action = get_random_1d_action(admissible_actions)
                action_idx = all_actions.index(action)
            else:
                pass
        else:
            pass
        action_desc = ActionDesc(
            action_type=ACT_TYPE.rnd_walk, action_idx=action_idx,
            token_idx=self.actor.action_matrix[action_idx],
            action_len=self.actor.action_len[action_idx],
            action=action)
        return action_desc

    def choose_action(
            self, actions: List[str], all_actions: List[str],
            actions_mask: bytes, instant_reward: float) -> ActionDesc:
        """
        Choose an action by
          1) try rule-based policy;
          2) try epsilon search learned policy;
          3) jitter go
        :param actions:
        :param all_actions:
        :param actions_mask:
        :param instant_reward:
        :return:
        """
        action_desc = self.rule_based_policy(
            actions, all_actions, instant_reward)
        if action_desc.action_idx is None:
            action_desc = self.random_walk_for_collecting_fp(
                actions, all_actions)
            if action_desc.action_idx is None:
                action_desc = self.get_an_eps_action(actions_mask)
                action_desc = self.jitter_go_action(
                    action_desc, actions, all_actions)
            else:
                pass
        else:
            pass
        return action_desc

    def get_instant_reward(
            self, score: float, master: str, is_terminal: bool,
            won: bool) -> float:
        # only penalize the final score if the agent choose a bad action.
        # do not penalize if failed because of out-of-steps.
        if is_terminal and not won and "you lost" in master:
            # self.info("game terminate and fail, final reward change"
            #           " from {} to -1".format(instant_reward))
            instant_reward = -1
        else:
            instant_reward = self.clip_reward(
                score - 0.1 - self._cumulative_score -
                self.negative_response_reward(master))
            if self.hp.use_step_wise_reward:
                if (master == self._prev_master
                        and self._last_action is not None
                        and self._last_action.action ==
                        self._prev_last_action and
                        instant_reward < 0):
                    instant_reward = self.clip_reward(
                        instant_reward + self._cumulative_penalty)
                    # self.debug("repeated bad try, decrease reward by {},"
                    #            " reward changed to {}".format(
                    #     self.prev_cumulative_penalty, instant_reward))
                    self._cumulative_penalty = self._cumulative_penalty - 0.1
                else:
                    self._prev_last_action = (
                        self._last_action.action
                        if self._last_action is not None else None)
                    self._prev_master = master
                    self._cumulative_penalty = -0.1
        self._cumulative_score = score
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

    def train_one_batch(self) -> None:
        """
        Train one batch of samples.
        Load target model if not exist, save current model when necessary.
        """
        if self.total_t == self.hp.observation_t:
            self.epoch_start_t = ctime()
        # if there is not a well-trained model, it is unreasonable
        # to use target model.
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)

        trajectory_id = [m[0].tid for m in b_memory]
        state_id = [m[0].sid for m in b_memory]
        action_id = [m[0].aid for m in b_memory]
        game_id = [m[0].gid for m in b_memory]
        reward = [m[0].reward for m in b_memory]
        is_terminal = [m[0].is_terminal for m in b_memory]
        action_mask = [m[0].action_mask for m in b_memory]
        next_action_mask = [m[0].next_action_mask for m in b_memory]

        pre_action_mask = self.from_bytes(action_mask)
        post_action_mask = self.from_bytes(next_action_mask)

        post_trajectories = self.tjs.fetch_batch_states(trajectory_id, state_id)
        pre_trajectories = self.tjs.fetch_batch_states(
            trajectory_id, [sid - 1 for sid in state_id])

        post_states = [
            state[0] for state in
            self.stc.fetch_batch_states(trajectory_id, state_id)]
        pre_states = [
            state[0] for state in self.stc.fetch_batch_states(
                trajectory_id, [sid - 1 for sid in state_id])]

        # make sure the p_states and s_states are in the same game.
        # otherwise, it won't make sense to use the same action matrix.
        action_len = (
            [self.actor.get_action_len(gid) for gid in game_id])
        max_action_len = np.max(action_len)
        action_matrix = (
            [self.actor.get_action_matrix(gid)[:, :max_action_len]
             for gid in game_id])

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
            step=self.total_t, others=None)

        self.memo.batch_update(b_idx, b_weight)

        if self.is_time_to_save():
            self.save_snapshot()
            self.core.save_model()
            self.core.create_or_reload_target_model()

    def _clean_stale_context(self, tids: List[int]) -> None:
        self.debug("tjs deletes {}".format(tids))
        self.tjs.request_delete_keys(tids)
        self.stc.request_delete_keys(tids)

    def update_status_impl(
            self, master: str, cleaned_obs: str, instant_reward: float,
            infos: Dict[str, List[Any]]) -> None:
        if self.hp.collect_floor_plan:
            self._curr_place = self.collect_floor_plan(master, self._prev_place)
        else:
            self._curr_place = None
        # use see cookbook again if gain one reward
        if instant_reward > 0:
            self._see_cookbook = False

    def get_admissible_actions(
            self, infos: Dict[str, List[Any]]) -> List[str]:
        return [a.lower() for a in infos[INFO_KEY.actions][0]]

    def update_status(
            self, obs: List[str], scores: List[float], dones: List[bool],
            infos: Dict[str, List[Any]]) -> Tuple[str, str, float]:
        self._prev_place = self._curr_place
        master = (self._get_master_starter(obs, infos)
                  if self.in_game_t == 0 else obs[0])
        if self.hp.apply_dependency_parser:
            cleaned_obs = self.dp.reorder(master)
        else:
            cleaned_obs = self.get_cleaned_master(master)

        if (self._last_action is not None
                and self._last_action.action == ACT.examine_cookbook):
            self._theme_words[self.game_id] = self.get_theme_words(master)
            self.debug(
                "get theme words: {}".format(self._theme_words[self.game_id]))

        instant_reward = self.get_instant_reward(
            scores[0], cleaned_obs, dones[0], infos[INFO_KEY.won][0])

        self.update_status_impl(master, cleaned_obs, instant_reward, infos)

        if 0 < self.in_game_t:  # pass the 1st master
            self.debug(
                "mode: {}, t: {}, in_game_t: {}, eps: {}, {}, master: {},"
                " reward: {}, raw_score: {}, is_terminal: {}".format(
                    self.mode(), self.total_t,
                    self.in_game_t, self.eps, self._last_action,
                    cleaned_obs, instant_reward, scores[0], dones[0]))
        elif self.in_game_t == 0:
            self.info(
                "mode: {}, master: {}, max_score: {}".format(
                    self.mode(), cleaned_obs, infos[INFO_KEY.max_score]))
        else:
            pass
        return master, cleaned_obs, instant_reward

    def collect_new_sample(
            self, master: str, instant_reward: float, dones: List[bool],
            infos: Dict[str, List[Any]]) -> Tuple[
            List[str], List[str], bytes, float]:

        self.tjs.append(ActionMaster(
            action=self._last_action if self._last_action else "",
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

        actions = self.get_admissible_actions(infos)
        actions = self.filter_admissible_actions(actions)
        actions = self.go_with_floor_plan(actions)
        actions_mask = self.actor.extend(actions)
        all_actions = self.actor.actions

        if self.is_training and self.tjs.get_last_sid() > 0:
            original_data = self.memo.append(DRRNMemo(
                tid=self.tjs.get_current_tid(),
                sid=self.tjs.get_last_sid(),
                gid=self.game_id,
                aid=self._last_action.action_idx,
                token_id=self._last_action.token_idx,
                a_len=self._last_action.action_len,
                reward=instant_reward,
                is_terminal=dones[0],
                action_mask=self._last_actions_mask,
                next_action_mask=actions_mask
            ))
            if isinstance(original_data, DRRNMemo):
                if original_data.is_terminal:
                    self._stale_tids.append(original_data.tid)

        return actions, all_actions, actions_mask, instant_reward

    def next_step_action(
            self, actions: List[str], all_actions: List[str],
            actions_mask: bytes, instant_reward: float) -> str:

        if self.is_training:
            self.eps = self.eps_getter.eps(self.total_t - self.hp.observation_t)
        else:
            pass
        self._last_action = self.choose_action(
            actions, all_actions, actions_mask, instant_reward)
        action = self._last_action.action
        action_idx = self._last_action.action_idx

        if self._last_action.action_type == ACT_TYPE.policy_drrn:
            self._cnt_action[action_idx] += 0.1

        self._last_actions_mask = actions_mask
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
        master, cleaned_obs, instant_reward = self.update_status(
            obs, scores, dones, infos)
        (actions, all_actions, actions_mask, instant_reward
         ) = self.collect_new_sample(master, instant_reward, dones, infos)
        # notice the position of all(dones)
        # make sure add the last action-master pair into memory
        if all(dones):
            self._end_episode(obs, scores, infos)
            return None

        player_t = self.next_step_action(
            actions, all_actions, actions_mask, instant_reward)
        if self.is_training and self.total_t >= self.hp.observation_t:
            self.train_one_batch()
        self.total_t += 1
        self.in_game_t += 1
        return [player_t] * len(obs)
