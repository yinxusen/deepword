import glob
import hashlib
import os
import re
from os import remove as prm
from os.path import join as pjoin
from typing import List, Dict, Any, Optional

import numpy as np
import tensorflow as tf
from bert.tokenization import FullTokenizer as BertTokenizer
from albert.tokenization import FullTokenizer as AlbertTokenizer
from bitarray import bitarray
from tensorflow.python.client import device_lib
from textworld import EnvInfos

from deeptextworld import trajectory
from deeptextworld.action import ActionCollector
from deeptextworld.floor_plan import FloorPlanCollector
from deeptextworld.hparams import save_hparams, output_hparams, copy_hparams
from deeptextworld.models.dqn_func import get_random_1Daction
from deeptextworld.tree_memory import TreeMemory
from deeptextworld.utils import ctime
from deeptextworld.agents.utils import *


class BaseAgent(Logging):
    """
    """

    def __init__(self, hp, model_dir):
        super(BaseAgent, self).__init__()
        self.model_dir = model_dir

        self.tjs_prefix = "trajectories"
        self.tjs_seg_prefix = "segmentation"
        self.action_prefix = "actions"
        self.memo_prefix = "memo"
        self.fp_prefix = "floor_plan"

        self.inv_direction = {
            ACT.gs: ACT.gn, ACT.gn: ACT.gs,
            ACT.ge: ACT.gw, ACT.gw: ACT.ge}

        self.hp, self.tokenizer = self.init_tokens(hp)

        self.info(output_hparams(self.hp))

        self.tjs = None
        self.tjs_seg = None
        self.memo = None
        self.model = None
        self.target_model = None
        self.actor = None  # action collector
        self.floor_plan = None
        self.dp = None
        self.loaded_ckpt_step = 0

        self._initialized = False
        self._episode_has_started = False
        self.total_t = 0
        self.in_game_t = 0
        # eps decaying test for all-tiers
        self.eps_getter = ScannerDecayEPS(
            decay_step=10000000, decay_range=1000000)
        # self.eps_getter = LinearDecayedEPS(
        #     decay_step=self.hp.annealing_eps_t,
        #     init_eps=self.hp.init_eps, final_eps=self.hp.final_eps)
        self.eps = 0
        self.sess = None
        self.target_sess = None
        self.is_training = True
        self.train_summary_writer = None
        self.ckpt_path = os.path.join(self.model_dir, 'last_weights')
        self.best_ckpt_path = os.path.join(self.model_dir, 'best_weights')
        self.ckpt_prefix = os.path.join(self.ckpt_path, 'after-epoch')
        self.best_ckpt_prefix = os.path.join(self.best_ckpt_path, 'after-epoch')
        self.saver = None
        self.target_saver = None
        self.snapshot_saved = False
        self.epoch_start_t = 0
        self.d4train, self.d4eval, self.d4target = self.init_devices()

        self._stale_tids = []

        self._last_actions_mask = None
        self._last_action = None

        self._cumulative_score = 0
        self._cumulative_penalty = -0.1
        self._prev_last_action = None
        self._prev_master = None
        self._prev_place = None
        self._curr_place = None

        self.game_id = None
        self._theme_words = {}

        self._see_cookbook = False
        self._cnt_action = None

        self._largest_valid_tag = 0
        self._stale_tags = None

        # self._action_recorder = {}
        # self._winning_recorder = {}
        # self._per_game_recorder = None
        # self._actions_to_remove = {}

    def set_d4eval(self, device):
        self.d4eval = device

    @classmethod
    def report_status(cls, lst_of_status):
        return ', '.join(
            map(lambda k_v: '{}: {}'.format(k_v[0], k_v[1]), lst_of_status))

    @classmethod
    def reverse_annealing_gamma(cls, init_gamma, final_gamma, t, total_t):
        gamma_t = init_gamma + ((final_gamma - init_gamma) * 1. / total_t) * t
        return min(gamma_t, final_gamma)

    @classmethod
    def annealing_eps(cls, init_eps, final_eps, t, total_t):
        eps_t = init_eps - ((init_eps - final_eps) * 1. / total_t) * t
        return max(eps_t, final_eps)

    @classmethod
    def from_bytes(cls, byte_action_masks):
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
    def count_trainable(cls, trainable_vars, mask=None):
        total_parameters = 0
        if mask is not None:
            if type(mask) is list:
                trainable_vars = filter(lambda v: v.op.name not in mask,
                                        trainable_vars)
            elif type(mask) is str:
                trainable_vars = filter(lambda v: v.op.name != mask,
                                        trainable_vars)
            else:
                pass
        else:
            pass
        for variable in trainable_vars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    @classmethod
    def get_theme_words(cls, recipe):
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
    def get_path_tags(cls, path, prefix):
        all_paths = glob.glob(os.path.join(path, "{}-*.npz".format(prefix)),
                              recursive=False)
        tags = map(lambda fn: int(os.path.splitext(fn)[0].split("-")[1]),
                   map(lambda p: os.path.basename(p), all_paths))
        return tags

    @classmethod
    def clip_reward(cls, reward):
        """clip reward into [-1, 1]"""
        return max(min(reward, 1), -1)

    @classmethod
    def contain_words(cls, sentence, words):
        return any(map(lambda w: w in sentence, words))

    @classmethod
    def get_room_name(cls, master):
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
    def negative_response_reward(cls, master):
        return 0

    @classmethod
    def get_hash(cls, txt):
        return hashlib.md5(txt.encode("utf-8")).hexdigest()

    def select_additional_infos(self):
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
    def init_tokens(cls, hp):
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

    def init_actions(self, hp, tokenizer, action_path, with_loading=True):
        action_collector = ActionCollector(
            tokenizer,
            hp.n_actions, hp.n_tokens_per_action,
            hp.unk_val_id, hp.padding_val_id, hp.eos_id, hp.pad_eos)
        if with_loading:
            try:
                action_collector.load_actions(action_path)
                # action_collector.extend(load_actions(hp.action_file))
            except IOError as e:
                self.info("load actions error: \n{}".format(e))
        return action_collector

    def init_trajectory(self, hp, tjs_path, with_loading=True):
        tjs_creator = getattr(trajectory, hp.tjs_creator)
        tjs = tjs_creator(hp, padding_val=hp.padding_val_id)
        if with_loading:
            try:
                tjs.load_tjs(tjs_path)
            except IOError as e:
                self.info("load trajectory error: \n{}".format(e))
        return tjs

    def init_memo(self, hp, memo_path, with_loading=True):
        memory = TreeMemory(capacity=hp.replay_mem)
        if with_loading:
            try:
                memory.load_memo(memo_path)
            except IOError as e:
                self.info("load memory error: \n{}".format(e))
        return memory

    def init_floor_plan(self, fp_path, with_loading=True):
        fp = FloorPlanCollector()
        if with_loading:
            try:
                fp.load_fps(fp_path)
            except IOError as e:
                self.info("load floor plan error: \n{}".format(e))
        return fp

    def tokenize(self, master):
        return ' '.join(self.tokenizer.tokenize(master))

    def index_string(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def zero_mask_bytes(self):
        """
        self.hp.n_actions should be in the format of 2**n
        :return:
        """
        bit_mask_vec = bitarray(self.hp.n_actions, endian="little")
        bit_mask_vec[::] = False
        bit_mask_vec[-1] = True  # to avoid tail trimming for bytes
        return bit_mask_vec.tobytes()

    def get_an_eps_action(self, action_mask):
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param action_mask:
        """
        raise NotImplementedError()

    def train(self):
        self.is_training = True
        self._init()

    def eval(self, load_best=True):
        self.is_training = False
        self._init(load_best)

    def reset(self, restore_from=None):
        """
        reset is only used for evaluation during training
        do not use it at anywhere else.
        """
        self.is_training = False
        self._initialized = False
        self._init(load_best=False, restore_from=restore_from)

    def safe_loading(self, model, sess, saver, restore_from):
        # Reload weights from directory if specified
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

    def create_model(self, is_training=True, device=None):
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

    def load_model(
            self, sess, model, saver, restore_from=None, load_best=False):
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

    def create_model_instance(self, device):
        raise NotImplementedError()

    def create_eval_model_instance(self, device):
        raise NotImplementedError()

    def train_impl(self, sess, t, summary_writer, target_sess, target_model):
        raise NotImplementedError()

    def _init(self, load_best=False, restore_from=None):
        """
        load actions, trajectories, memory, model, etc.
        """
        if self._initialized:
            self.error("the agent was initialized")
            return
        self._init_impl(load_best, restore_from)
        self._initialized = True

    def _get_context_obj_path_w_tag(self, prefix, tag):
        return pjoin(
            self.model_dir, "{}-{}.npz".format(prefix, tag))

    def _get_context_obj_path(self, prefix):
        return self._get_context_obj_path_w_tag(prefix, self._largest_valid_tag)

    def _get_context_obj_new_path(self, prefix):
        return self._get_context_obj_path_w_tag(prefix, self.total_t)

    def _load_context_objs(self):
        valid_tags = self.get_compatible_snapshot_tag()
        self._largest_valid_tag = max(valid_tags) if len(valid_tags) != 0 else 0
        self.info("try to load from tag: {}".format(self._largest_valid_tag))

        action_path = self._get_context_obj_path(self.action_prefix)
        tjs_path = self._get_context_obj_path(self.tjs_prefix)
        tjs_seg_path = self._get_context_obj_path(self.tjs_seg_prefix)
        memo_path = self._get_context_obj_path(self.memo_prefix)
        fp_path = self._get_context_obj_path(self.fp_prefix)

        # always loading actions to avoid different action index for DQN
        self.actor = self.init_actions(
            self.hp, self.tokenizer, action_path,
            with_loading=self.is_training)
        self.tjs = self.init_trajectory(
            self.hp, tjs_path, with_loading=self.is_training)
        self.tjs_seg = self.init_trajectory(
            self.hp, tjs_seg_path, with_loading=self.is_training)
        self.memo = self.init_memo(
            self.hp, memo_path, with_loading=self.is_training)
        self.floor_plan = self.init_floor_plan(
            fp_path, with_loading=self.is_training)

    def _init_impl(self, load_best=False, restore_from=None):
        self._load_context_objs()
        if self.model is None:
            self.sess, self.model, self.saver = self.create_model(
                self.is_training)
        self.loaded_ckpt_step = self.load_model(
            self.sess, self.model, self.saver, restore_from, load_best)

        if self.is_training:
            if self.loaded_ckpt_step > 0:
                self._create_n_load_target_model(restore_from, load_best)
            self.eps = self.hp.init_eps
            train_summary_dir = os.path.join(
                self.model_dir, "summaries", "train")
            self.train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, self.sess.graph)
            if self.hp.start_t_ignore_model_t:
                self.total_t = min(
                    self.hp.observation_t,
                    len(self.memo) if self.memo is not None else 0)
            else:
                self.total_t = self.loaded_ckpt_step + self.hp.observation_t
        else:
            self.eps = 0
            self.total_t = 0

    def _get_master_starter(self, obs, infos):
        assert INFO_KEY.desc in infos, "request description is required"
        assert INFO_KEY.inventory in infos, "request inventory is required"
        return "{}\n{}".format(
            infos[INFO_KEY.desc][0], infos[INFO_KEY.inventory][0])

    def _start_episode(self, obs, infos):
        """
        Prepare the agent for the upcoming episode.
        :param obs: initial feedback from each game
        :param infos: additional infors of each game
        :return:
        """
        if not self._initialized:
            self._init()
        self._start_episode_impl(obs, infos)

    def _start_episode_impl(self, obs, infos):
        self.tjs.add_new_tj()
        self.tjs_seg.add_new_tj(tid=self.tjs.get_current_tid())
        master_starter = self._get_master_starter(obs, infos)
        self.game_id = self.get_hash(master_starter)
        self.actor.add_new_episode(eid=self.game_id)
        self.floor_plan.add_new_episode(eid=self.game_id)
        self.in_game_t = 0
        self._cumulative_score = 0
        self._episode_has_started = True
        self._prev_place = None
        self._curr_place = None
        self._cnt_action = np.zeros(self.hp.n_actions)
        # if self.game_id not in self._action_recorder:
        #     self._action_recorder[self.game_id] = []
        # if self.game_id not in self._winning_recorder:
        #     self._winning_recorder[self.game_id] = False
        # if self.game_id not in self._actions_to_remove:
        #     self._actions_to_remove[self.game_id] = set()
        if self.game_id not in self._theme_words:
            self._theme_words[self.game_id] = []
        self._per_game_recorder = []
        self._see_cookbook = False
        self.debug("infos: {}".format(infos))
        # self._theme_words[self.game_id] = self.get_theme_words(
        #     infos[INFO_KEY.recipe][0])

    def mode(self):
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
        # if len(self.memo) > 2 * self.hp.replay_mem:
        #     to_delete_tj_id = self.memo.clear_old_memory()
        #     self.tjs.request_delete_key(to_delete_tj_id)
        self.info(
            "mode: {}, obs: {}, #step: {}, score: {}, won: {},"
            " last_eps: {}".format(
                self.mode(), obs[0], self.in_game_t, scores[0],
                infos[INFO_KEY.won], self.eps))
        # TODO: make clear of what need to clean before & after an episode.
        # self._winning_recorder[self.game_id] = infos[K_HAS_WON][0]
        # self._action_recorder[self.game_id] = self._per_game_recorder
        # if ((not infos[K_HAS_WON][0]) and
        #         (0 < len(self._per_game_recorder) < 100)):
        #     if (self._per_game_recorder[-1] not in
        #             self._per_game_recorder[:-1]):
        #         self._actions_to_remove[self.game_id].add(
        #             self._per_game_recorder[-1])
        #     else:
        #         pass  # repeat dangerous actions
        # self.debug("actions to remove {} for game {}".format(
        #     self._actions_to_remove[self.game_id], self.game_id))
        self._episode_has_started = False
        self._last_actions_mask = None
        self.game_id = None
        self._last_action = None
        self._cumulative_penalty = -0.1
        self._prev_last_action = None
        self._prev_master = None

    def save_best_model(self):
        self.info("save the best model so far")
        self.saver.save(
            self.sess, self.best_ckpt_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=self.model.graph))
        self.info("the best model saved")

    def _delete_stale_context_objs(self):
        valid_tags = self.get_compatible_snapshot_tag()
        if len(valid_tags) > self.hp.max_snapshot_to_keep:
            self._stale_tags = list(reversed(sorted(
                valid_tags)))[self.hp.max_snapshot_to_keep:]
            self.info("tags to be deleted: {}".format(self._stale_tags))
            for tag in self._stale_tags:
                prm(self._get_context_obj_path_w_tag(self.memo_prefix, tag))
                prm(self._get_context_obj_path_w_tag(self.tjs_prefix, tag))
                prm(self._get_context_obj_path_w_tag(self.tjs_seg_prefix, tag))
                prm(self._get_context_obj_path_w_tag(self.action_prefix, tag))
                prm(self._get_context_obj_path_w_tag(self.fp_prefix, tag))

    def _save_context_objs(self):
        action_path = self._get_context_obj_new_path(self.action_prefix)
        tjs_path = self._get_context_obj_new_path(self.tjs_prefix)
        tjs_seg_path = self._get_context_obj_new_path(self.tjs_seg_prefix)
        memo_path = self._get_context_obj_new_path(self.memo_prefix)
        fp_path = self._get_context_obj_new_path(self.fp_prefix)

        self.memo.save_memo(memo_path)
        self.tjs.save_tjs(tjs_path)
        self.tjs_seg.save_tjs(tjs_seg_path)
        self.actor.save_actions(action_path)
        self.floor_plan.save_fps(fp_path)

    def save_snapshot(self):
        self.info('save model')
        self.saver.save(
            self.sess, self.ckpt_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=self.model.graph))
        self.info('save snapshot of the agent')
        self._save_context_objs()
        self._delete_stale_context_objs()
        self._clean_stale_context(self._stale_tids)
        # notice that we should not save hparams when evaluating
        # that's why I move this function calling here from __init__
        save_hparams(
            self.hp, pjoin(self.model_dir, 'hparams.json'),
            use_relative_path=True)

    def get_compatible_snapshot_tag(self):
        action_tags = self.get_path_tags(self.model_dir, self.action_prefix)
        memo_tags = self.get_path_tags(self.model_dir, self.memo_prefix)
        tjs_tags = self.get_path_tags(self.model_dir, self.tjs_prefix)
        tjs_seg_tags = self.get_path_tags(self.model_dir, self.tjs_seg_prefix)
        fp_tags = self.get_path_tags(self.model_dir, self.fp_prefix)

        valid_tags = set(action_tags)
        valid_tags.intersection_update(memo_tags)
        valid_tags.intersection_update(tjs_tags)
        valid_tags.intersection_update(tjs_seg_tags)
        valid_tags.intersection_update(fp_tags)

        return list(valid_tags)

    def time_to_save(self):
        trained_steps = self.total_t - self.hp.observation_t + 1
        return (trained_steps % self.hp.save_gap_t == 0) and (trained_steps > 0)

    def contain_theme_words(self, actions):
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

    # def filter_admissible_actions(self, admissible_actions):
    #     actions = list(filter(
    #         lambda c: not c.startswith("examine"), admissible_actions))
    #     return list(set(actions))

    def filter_admissible_actions(self, admissible_actions):
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
        # if not self.is_training:
        #     if ((self._winning_recorder[self.game_id] is not None) and
        #             (not self._winning_recorder[self.game_id])):
        #         for a2remove in self._actions_to_remove[self.game_id]:
        #             try:
        #                 actions.remove(a2remove)
        #                 self.debug(
        #                     "action {} is removed".format(
        #                         a2remove))
        #             except ValueError as _:
        #                 self.debug(
        #                     "action {} is not found when remove".format(
        #                         a2remove))
        #     else:
        #         pass
        return actions

    def go_with_floor_plan(self, actions):
        local_map = self.floor_plan.get_map(self._curr_place)
        return (["{} to {}".format(a, local_map.get(a))
                 if a in local_map else a for a in actions])

    # def rule_based_policy(self, actions, all_actions, instant_reward):
    #     action_desc = ActionDesc(
    #         action_type=ACT_TYPE_RULE, action_idx=None,
    #         token_idx=None,
    #         action_len=None,
    #         action=None)
    #     return action_desc

    def rule_based_policy(self, actions, all_actions, instant_reward):
        # use hard set actions in the beginning and the end of one episode
        # if ((not self.is_training) and
        #         (self._winning_recorder[self.game_id] is not None) and
        #         self._winning_recorder[self.game_id]):
        #     try:
        #         action = self._action_recorder[self.game_id][self.in_game_t]
        #     except IndexError as _:
        #         self.debug("same game ID for different games error")
        #         action = None
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

    def _jitter_go_condition(self, action_desc, admissible_go_actions):
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
            self, prev_action_desc, actions, all_actions):
        action_desc = None
        admissible_go_actions = list(
            filter(lambda a: a.startswith("go"), actions))
        if self._jitter_go_condition(prev_action_desc, admissible_go_actions):
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

    def random_walk_for_collecting_fp(self, actions, all_actions):
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
                _, action = get_random_1Daction(admissible_actions)
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
            self, actions, all_actions, actions_mask, instant_reward):
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

    def get_instant_reward(self, score, master, is_terminal, won):
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

    def collect_floor_plan(self, master, prev_place):
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

    def train_one_batch(self):
        """
        Train one batch of samples.
        Load target model if not exist, save current model when necessary.
        """
        if self.total_t == self.hp.observation_t:
            self.epoch_start_t = ctime()
        # if there is not a well-trained model, it is unreasonable
        # to use target model.
        self.train_impl(
            self.sess, self.total_t, self.train_summary_writer,
            self.target_sess if self.target_sess else self.sess,
            self.target_model if self.target_model else self.model)
        self._save_agent_n_reload_target()

    def _create_n_load_target_model(self, restore_from, load_best):
        if self.target_sess is None:
            self.debug("create target model ...")
            (self.target_sess, self.target_model, self.target_saver
             ) = self.create_model(is_training=False, device=self.d4target)
        else:
            pass
        trained_step = self.load_model(
            self.target_sess, self.target_model, self.target_saver,
            restore_from, load_best)
        self.debug(
            "load target model from trained step {}".format(trained_step))

    def _save_agent_n_reload_target(self):
        if self.time_to_save():
            epoch_end_t = ctime()
            delta_time = epoch_end_t - self.epoch_start_t
            self.info('current epoch end')
            reports_time = [
                ('epoch time', delta_time),
                ('#batches per epoch', self.hp.save_gap_t),
                ('avg step time',
                 delta_time * 1.0 / self.hp.save_gap_t)]
            self.info(self.report_status(reports_time))
            self.save_snapshot()
            self._create_n_load_target_model(restore_from=None, load_best=False)
            self.snapshot_saved = True
            self.info("snapshot saved, ready for evaluation")
            self.epoch_start_t = ctime()

    def _clean_stale_context(self, tids):
        self.debug("tjs deletes {}".format(tids))
        self.tjs.request_delete_keys(tids)
        self.tjs_seg.request_delete_keys(tids)

    def feed_memory(
            self, instant_reward, is_terminal, action_mask, next_action_mask):
        original_data = self.memo.append(DRRNMemo(
            tid=self.tjs.get_current_tid(), sid=self.tjs.get_last_sid(),
            gid=self.game_id, aid=self._last_action.action_idx,
            token_id=self._last_action.token_idx,
            a_len=self._last_action.action_len,
            reward=instant_reward, is_terminal=is_terminal,
            action_mask=action_mask, next_action_mask=next_action_mask
        ))
        if isinstance(original_data, DRRNMemo):
            if original_data.is_terminal:
                self._stale_tids.append(original_data.tid)
                # self._clean_stale_context(original_data.tid)

    def update_status_impl(self, master, cleaned_obs, instant_reward, infos):
        if self.hp.collect_floor_plan:
            self._curr_place = self.collect_floor_plan(master, self._prev_place)
        else:
            self._curr_place = None
        # use see cookbook again if gain one reward
        if instant_reward > 0:
            self._see_cookbook = False

    def get_admissible_actions(self, infos=None):
        assert infos is not None and INFO_KEY.actions in infos
        return [a.lower() for a in infos[INFO_KEY.actions][0]]

    def update_status(self, obs, scores, dones, infos):
        self._prev_place = self._curr_place
        master = (self._get_master_starter(obs, infos)
                  if self.in_game_t == 0 else obs[0])
        if self.hp.apply_dependency_parser:
            cleaned_obs = self.dp.reorder(master)
        else:
            cleaned_obs = self.tokenize(master)

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
        return cleaned_obs, instant_reward

    def collect_new_sample(self, cleaned_obs, instant_reward, dones, infos):
        obs_idx = self.index_string(cleaned_obs.split())
        if self.in_game_t == 0 and self._last_action is None:
            act_idx = []
        else:
            act_idx = list(
                self._last_action.token_idx[:self._last_action.action_len])
        self.tjs.append(act_idx + obs_idx)
        self.tjs_seg.append([1] * len(act_idx) + [0] * len(obs_idx))

        actions = self.get_admissible_actions(infos)
        actions = self.filter_admissible_actions(actions)
        actions = self.go_with_floor_plan(actions)
        # self.info("admissible actions: {}".format(", ".join(sorted(actions))))
        actions_mask = self.actor.extend(actions)
        all_actions = self.actor.actions

        # make sure appending tjs first, otherwise the judgement could be wrong
        # pass the 1st master
        if self.is_training and self.tjs.get_last_sid() > 0:
            self.feed_memory(
                instant_reward, dones[0],
                self._last_actions_mask, actions_mask)
        else:
            pass

        return actions, all_actions, actions_mask, instant_reward

    def next_step_action(
            self, actions, all_actions, actions_mask, instant_reward):
        if self.is_training:
            self.eps = self.eps_getter.eps(self.total_t - self.hp.observation_t)
        else:
            pass
        self._last_action = self.choose_action(
            actions, all_actions, actions_mask, instant_reward)
        action = self._last_action.action
        action_idx = self._last_action.action_idx

        # self._per_game_recorder.append(action)

        if self._last_action.action_type == ACT_TYPE.policy_drrn:
            self._cnt_action[action_idx] += 0.1
        else:
            # self.debug("cnt action ignore hard_set_action")
            pass
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
        :return:

        Notes:
            Commands returned for games marked as `done` have no effect.
            The states for finished games are simply copy over until all
            games are done.
        """
        if not self._episode_has_started:
            self._start_episode(obs, infos)

        assert len(obs) == 1, "cannot handle batch game training"
        cleaned_obs, instant_reward = self.update_status(
            obs, scores, dones, infos)
        (actions, all_actions, actions_mask, instant_reward
         ) = self.collect_new_sample(cleaned_obs, instant_reward, dones, infos)
        # notice the position of all(dones)
        # make sure add the last action-master pair into memory
        if all(dones):
            self._end_episode(obs, scores, infos)
            return  # Nothing to return.
        player_t = self.next_step_action(
            actions, all_actions, actions_mask, instant_reward)
        if self.is_training and self.total_t >= self.hp.observation_t:
            self.train_one_batch()
        self.total_t += 1
        self.in_game_t += 1
        return [player_t] * len(obs)
