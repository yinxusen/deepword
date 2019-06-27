import glob
import hashlib
import os
import random
import re
from typing import List, Dict, Any, Optional

import collections
import numpy as np
import tensorflow as tf
from bitarray import bitarray
from nltk import word_tokenize, sent_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser

from deeptextworld import trajectory
from deeptextworld.action import ActionCollector
from deeptextworld.dqn_func import get_random_1Daction
from deeptextworld.floor_plan import FloorPlanCollector
from deeptextworld.hparams import save_hparams, output_hparams, copy_hparams
from deeptextworld.log import Logging
from deeptextworld.tree_memory import TreeMemory
from deeptextworld.utils import get_token2idx, load_lower_vocab, load_actions, \
    ctime


class DRRNMemo(collections.namedtuple(
    "DRRNMemo",
    ("tid", "sid", "gid", "aid", "reward", "is_terminal",
     "action_mask", "next_action_mask"))):
    pass


class ActionDesc(collections.namedtuple(
    "ActionDesc",
    ("action_type", "action_idx", "action"))):
    def __repr__(self):
        return "action_type: {}, action_idx: {}, action: {}".format(
            self.action_type, self.action_idx, self.action)


class DependencyParserReorder(Logging):
    """
    Use dependency parser to reorder master sentences.
    Make sure to open Stanford CoreNLP server first.
    """
    def __init__(self, padding_val, stride_len):
        super(DependencyParserReorder, self).__init__()
        # be sure of starting CoreNLP server first
        self.parser = CoreNLPDependencyParser()
        # use dict to avoid parse the same sentences.
        self.parsed_sentences = dict()
        self.sep_sent = (" " + " ".join([padding_val] * stride_len)
                         + " ")

    def reorder_sent(self, sent):
        tree = next(self.parser.raw_parse(sent)).tree()
        t_labels = ([
            [head.label()] +
            [child if type(child) is str else child.label() for child in head]
            for head in tree.subtrees()])
        t_str = [" ".join(labels) for labels in t_labels]
        return self.sep_sent.join(t_str)

    def reorder_block(self, master):
        """
        Notice that four padding letters " O O O O " can only work up to 5-gram
        :param master:
        :return:
        """
        sent_list = list(filter(lambda sent: sent != "", sent_tokenize(master)))
        tree_strs = []
        for s in sent_list:
            if not s in self.parsed_sentences:
                t_str = self.reorder_sent(s)
                self.parsed_sentences[s] = t_str
                self.info("parse {} into {}".format(s, t_str))
            else:
                self.info("found parsed {}".format(s))
            tree_strs.append(self.parsed_sentences[s])
        return self.sep_sent.join(tree_strs)

    def reorder(self, master):
        if master == "":
            return master
        lines = map(lambda l: l.lower(),
                    filter(lambda l: l.strip() != "", master.split("\n")))
        reordered_lines = map(lambda l: self.reorder_block(l), lines)
        return self.sep_sent + self.sep_sent.join(reordered_lines) + self.sep_sent


ACT_EXAMINE_COOKBOOK = "examine cookbook"
ACT_PREPARE_MEAL = "prepare meal"
# ACT_EXAMINE_COOKBOOK = ACT_PREPARE_MEAL
ACT_EAT_MEAL = "eat meal"
ACT_LOOK = "look"
ACT_INVENTORY = "inventory"
ACT_GN = "go north"
ACT_GS = "go south"
ACT_GE = "go east"
ACT_GW = "go west"

ACT_TYPE_RND_CHOOSE = "random_choose_action"
ACT_TYPE_RULE = "rule_based_action"
ACT_TYPE_RND_WALK = "random_walk_action"
ACT_TYPE_NN = "learned_action"
ACT_TYPE_JITTER = "jitter_action"


class BaseAgent(Logging):
    """
    """
    def __init__(self, hp, model_dir):
        super(BaseAgent, self).__init__()
        self.model_dir = model_dir

        self.tjs_prefix = "trajectories"
        self.action_prefix = "actions"
        self.memo_prefix = "memo"
        self.fp_prefix = "floor_plan"

        self.inv_direction = {
            ACT_GS: ACT_GN, ACT_GN: ACT_GS,
            ACT_GE: ACT_GW, ACT_GW: ACT_GE}

        self.hp, self.tokens, self.token2idx = self.init_tokens(hp)
        self.info(output_hparams(self.hp))

        self.tjs = None
        self.memo = None
        self.model = None
        self.target_model = None
        self.action_collector = None
        self.floor_plan = None
        self.dp = None

        self._initialized = False
        self._episode_has_started = False
        self.total_t = 0
        self.in_game_t = 0
        self.eps = 0
        self.sess = None
        self.target_sess = None
        self.is_training = True
        self.train_summary_writer = None
        self.chkp_path = os.path.join(self.model_dir, 'last_weights')
        self.best_chkp_path = os.path.join(self.model_dir, 'best_weights')
        self.chkp_prefix = os.path.join(self.chkp_path, 'after-epoch')
        self.best_chkp_prefix = os.path.join(self.best_chkp_path, 'after-epoch')
        self.saver = None
        self.target_saver = None
        self.snapshot_saved = False
        self.epoch_start_t = 0

        self._last_actions_mask = None
        self._last_action_desc = None

        self._cumulative_score = 0
        self._cumulative_penalty = -0.1
        self._prev_last_action = None
        self._prev_master = None
        self._prev_place = None

        self.game_id = None
        self._theme_words = None
        self._inventory = None
        self.__obs = None
        self.__see_cookbook = False
        self._cnt_action = None

        self.__action_recorder = None
        self.__winning_recorder = None
        self.__per_game_recorder = None
        self.__actions_to_remove = None

    def init_tokens(self, hp):
        """
        :param hp:
        :return:
        """
        new_hp = copy_hparams(hp)
        # make sure that padding_val is indexed as 0.
        additional_tokens = [hp.padding_val, hp.unk_val, hp.sos, hp.eos]
        tokens = additional_tokens + list(load_lower_vocab(hp.vocab_file))
        token2idx = get_token2idx(tokens)
        new_hp.set_hparam('vocab_size', len(tokens))
        new_hp.set_hparam('sos_id', token2idx[hp.sos])
        new_hp.set_hparam('eos_id', token2idx[hp.eos])
        new_hp.set_hparam('padding_val_id', token2idx[hp.padding_val])
        new_hp.set_hparam('unk_val_id', token2idx[hp.unk_val])
        return new_hp, tokens, token2idx

    def init_actions(self, hp, token2idx, action_path, with_loading=True):
        action_collector = ActionCollector(
            hp.n_actions, hp.n_tokens_per_action, token2idx,
            hp.unk_val_id, hp.padding_val_id)
        if with_loading:
            try:
                action_collector.load_actions(action_path)
                action_collector.extend(load_actions(hp.action_file))
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

    def _padding_lines(self, sents):
        """
        add padding sentence between lines
        """
        padding_sent = " {} ".format(" ".join([self.hp.padding_val] * 4))
        padded = padding_sent + padding_sent.join(sents) + padding_sent
        return padded

    def tokenize(self, master):
        """
        Tokenize and lowercase master. A space-chained tokens will be returned.
        # TODO: sentences that are tokenized cannot use tokenize again.
        """
        sents = sent_tokenize(master)
        tokenized = map(
            lambda s: ' '.join([t.lower() for t in word_tokenize(s)]),
            sents)
        if self.hp.use_padding_over_lines:
            return self._padding_lines(tokenized)
        else:
            return " ".join(tokenized)

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

    def index_string(self, sentence):
        indexed = [self.token2idx.get(t, self.hp.unk_val_id) for t in sentence]
        return indexed

    @classmethod
    def fromBytes(cls, action_mask):
        retval = []
        for mask in action_mask:
           bit_mask = bitarray(endian='little')
           bit_mask.frombytes(mask)
           bit_mask[-1] = False
           retval.append(bit_mask.tolist())
        return np.asarray(retval, dtype=np.int32)

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

    def reset(self):
        self._initialized = False
        self._init()

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

    def create_n_load_model(
            self, placement="/device:GPU:0",
            load_best=False, is_training=True):
        start_t = 0
        with tf.device(placement):
            if is_training:
                model = self.create_model_instance()
                self.info("create train model")
            else:
                model = self.create_eval_model_instance()
                self.info("create eval model")
        train_conf = tf.ConfigProto(log_device_placement=False,
                                    allow_soft_placement=True)
        train_sess = tf.Session(graph=model.graph, config=train_conf)
        with model.graph.as_default():
            train_sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=self.hp.max_snapshot_to_keep,
                                   save_relative_paths=True)
            if load_best:
                restore_from = tf.train.latest_checkpoint(self.best_chkp_path)
            else:
                restore_from = tf.train.latest_checkpoint(self.chkp_path)

            if restore_from is not None:
                # Reload weights from directory if specified
                self.info(
                    "Try to restore parameters from: {}".format(restore_from))
                saver.restore(train_sess, restore_from)
                if not self.hp.start_t_ignore_model_t:
                    global_step = tf.train.get_or_create_global_step()
                    trained_steps = train_sess.run(global_step)
                    start_t = trained_steps + self.hp.observation_t
            else:
                self.info('No checkpoint to load, training from scratch')
        return train_sess, start_t, saver, model

    def create_model_instance(self):
        raise NotImplementedError()

    def create_eval_model_instance(self):
        raise NotImplementedError()

    def train_impl(self, sess, t, summary_writer, target_sess):
        raise NotImplementedError()

    def _init(self, load_best=False):
        """
        load actions, trajectories, memory, model, etc.
        """
        if self._initialized:
            self.error("the agent was initialized")
            return

        if self.hp.apply_dependency_parser:
            # TODO: stride_len is hard fix here for maximum n-gram filter size 5
            self.dp = DependencyParserReorder(
                padding_val=self.hp.padding_val, stride_len=4)

        valid_tags = self.get_compatible_snapshot_tag()
        largest_valid_tag = max(valid_tags) if len(valid_tags) != 0 else 0
        self.info("try to load from tag: {}".format(largest_valid_tag))

        action_path = os.path.join(
            self.model_dir,
            "{}-{}.npz".format(self.action_prefix, largest_valid_tag))
        tjs_path = os.path.join(
            self.model_dir,
            "{}-{}.npz".format(self.tjs_prefix, largest_valid_tag))
        memo_path = os.path.join(
            self.model_dir,
            "{}-{}.npz".format(self.memo_prefix, largest_valid_tag))
        fp_path = os.path.join(
            self.model_dir,
            "{}-{}.npz".format(self.fp_prefix, largest_valid_tag))

        # always loading actions to avoid different action index for DQN
        self.action_collector = self.init_actions(
           self.hp, self.token2idx, action_path,
            with_loading=True)
        self.tjs = self.init_trajectory(
            self.hp, tjs_path, with_loading=self.is_training)
        self.memo = self.init_memo(
            self.hp, memo_path, with_loading=self.is_training)
        self.floor_plan = self.init_floor_plan(
            fp_path, with_loading=self.is_training)
        if self.is_training:
            self.sess, self.total_t, self.saver, self.model =\
                self.create_n_load_model()
            self.eps = self.hp.init_eps
            train_summary_dir = os.path.join(
                self.model_dir, "summaries", "train")
            self.train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, self.sess.graph)
        else:
            self.sess, _, self.saver, self.model = self.create_n_load_model(
                placement="/device:GPU:1", load_best=load_best,
                is_training=self.is_training)
            self.eps = 0
            self.total_t = 0
        self.__action_recorder = {}
        self.__actions_to_remove = {}
        self.__winning_recorder = {}
        self._initialized = True

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
        self.tjs.add_new_tj()
        # recipe = infos["extra.recipe"]
        # use stronger game identity
        # master_without_logo = infos["description"][0].replace("\n", " ")
        master = obs[0]
        self.game_id = hashlib.md5(master.encode("utf-8")).hexdigest()
        self.action_collector.add_new_episode(eid=self.game_id)
        self.floor_plan.add_new_episode(eid=self.game_id)
        self.in_game_t = 0
        self._cumulative_score = 0
        self._episode_has_started = True
        self._prev_place = None
        self._cnt_action = np.zeros(self.hp.n_actions)
        if self.game_id not in self.__action_recorder:
            self.__action_recorder[self.game_id] = None
            self.__winning_recorder[self.game_id] = None
            self.__actions_to_remove[self.game_id] = set()
        self.__per_game_recorder = []
        self.__see_cookbook = False
        self._theme_words = None
        self._inventory = []
        self.__obs = None
        # self.get_theme_words(recipe[0])

    def get_theme_words(self, recipe):
        theme_regex = ".*Ingredients:<\|>(.*)<\|>Directions.*"
        theme_words_search = re.search(
            theme_regex, recipe.replace("\n", "<|>"))
        if theme_words_search:
            theme_words = theme_words_search.group(1)
            self._theme_words = list(
                filter(lambda w: w != "",
                       map(lambda w: w.strip(), theme_words.split("<|>"))))
            self.debug("theme words: {}".format(", ".join(self._theme_words)))
        else:
            self.debug("no recipe found")

    def get_inventory(self, inventory_list):
        items = list(filter(
            lambda s: len(s) != 0,
            map(lambda s: s.strip(), inventory_list.split("\n"))))
        if len(items) > 1:
            # remove the first a/an/the/some ...
            items = list(map(lambda i: " ".join(i.split()[1:]), items[1:]))
        else:
            items = []
        self._inventory = items

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
        self.info("mode: {}, #step: {}, score: {}, has_won: {}".format(
            self.mode(), self.in_game_t, scores[0], infos["has_won"]))
        # TODO: make clear of what need to clean before & after an episode.
        self.__winning_recorder[self.game_id] = infos["has_won"][0]
        self.__action_recorder[self.game_id] = self.__per_game_recorder
        if ((not infos["has_won"][0]) and
                (0 < len(self.__per_game_recorder) < 100)):
            if (self.__per_game_recorder[-1] not in
                    self.__per_game_recorder[:-1]):
                self.__actions_to_remove[self.game_id].add(
                    self.__per_game_recorder[-1])
            else:
                pass  # repeat dangerous actions
        self.debug("actions to remove {} for game {}".format(
            self.__actions_to_remove[self.game_id], self.game_id))
        self._episode_has_started = False
        self._last_actions_mask = None
        self.game_id = None
        self._last_action_desc = None
        self._cumulative_penalty = -0.1
        self._prev_last_action = None
        self._prev_master = None

    def save_best_model(self):
        self.info("save the best model so far")
        self.saver.save(
            self.sess, self.best_chkp_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=self.model.graph))
        self.info("the best model saved")

    def save_snapshot(self):
        self.info('save model')
        self.saver.save(
            self.sess, self.chkp_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=self.model.graph))
        self.info('save snapshot of the agent')

        action_path = os.path.join(
            self.model_dir,
            "{}-{}.npz".format(self.action_prefix, self.total_t))
        tjs_path = os.path.join(
            self.model_dir,
            "{}-{}.npz".format(self.tjs_prefix, self.total_t))
        memo_path = os.path.join(
            self.model_dir,
            "{}-{}.npz".format(self.memo_prefix, self.total_t))
        fp_path = os.path.join(
            self.model_dir,
            "{}-{}.npz".format(self.fp_prefix, self.total_t))

        self.memo.save_memo(memo_path)
        self.tjs.save_tjs(tjs_path)
        self.action_collector.save_actions(action_path)
        self.floor_plan.save_fps(fp_path)

        valid_tags = self.get_compatible_snapshot_tag()
        if len(valid_tags) > self.hp.max_snapshot_to_keep:
            to_delete_tags = list(reversed(sorted(
                valid_tags)))[self.hp.max_snapshot_to_keep:]
            self.info("tags to be deleted: {}".format(to_delete_tags))
            for tag in to_delete_tags:
                os.remove(os.path.join(
                    self.model_dir,
                    "{}-{}.npz".format(self.memo_prefix, tag)))
                os.remove(os.path.join(
                    self.model_dir,
                    "{}-{}.npz".format(self.tjs_prefix, tag)))
                os.remove(os.path.join(
                    self.model_dir,
                    "{}-{}.npz".format(self.action_prefix, tag)))
                os.remove(os.path.join(
                    self.model_dir,
                    "{}-{}.npz".format(self.fp_prefix, tag)))
        # notice that we should not save hparams when evaluating
        # that's why I move this function calling here from __init__
        save_hparams(self.hp,
                     os.path.join(self.model_dir, 'hparams.json'),
                     use_relative_path=True)

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

    def get_compatible_snapshot_tag(self):
        action_tags = self.get_path_tags(self.model_dir, self.action_prefix)
        memo_tags = self.get_path_tags(self.model_dir, self.memo_prefix)
        tjs_tags = self.get_path_tags(self.model_dir, self.tjs_prefix)

        valid_tags = set(action_tags)
        valid_tags.intersection_update(memo_tags)
        valid_tags.intersection_update(tjs_tags)

        return list(valid_tags)

    def time_to_save(self):
        trained_steps = self.total_t - self.hp.observation_t + 1
        return (trained_steps % self.hp.save_gap_t == 0) and (trained_steps > 0)

    @classmethod
    def contain_words(cls, sentence, words):
        return any(map(lambda w: w in sentence, words))

    def contain_theme_words(self, actions):
        if self._theme_words is None:
            self.debug("no theme word found, use all actions")
            return actions, []
        contained = []
        others = []
        for a in actions:
            if self.contain_words(a, self._theme_words):
                contained.append(a)
            else:
                others.append(a)

        return contained, others

    @classmethod
    def get_room_name(cls, master):
        """
        Extract and lower room name.
        Return None if not exist.
        :param master:
        :return:
        """
        room_regex = "^\s*-= (.*) =-.*"
        room_search = re.search(room_regex, master)
        if room_search is not None:
            room_name = room_search.group(1).lower()
        else:
            room_name = None
        return room_name

    def filter_admissible_actions(self, admissible_actions, curr_place=None):
        """
        Filter unnecessary actions.
        :param admissible_actions:
        :return:
        """
        contained, others = self.contain_theme_words(admissible_actions)
        actions = [ACT_INVENTORY, ACT_LOOK] + contained
        actions = list(filter(lambda c: not c.startswith("examine"), actions))
        actions = list(filter(lambda c: not c.startswith("close"), actions))
        actions = list(filter(lambda c: not c.startswith("insert"), actions))
        actions = list(filter(lambda c: not c.startswith("eat"), actions))
        # don't drop useful ingredients if not in kitchen
        if (curr_place != "kitchen") or (not self.hp.drop_w_theme_words):
            actions = list(filter(lambda c: not c.startswith("drop"), actions))
        actions = list(filter(lambda c: not c.startswith("put"), actions))
        other_valid_commands = {
            ACT_PREPARE_MEAL, ACT_EAT_MEAL, ACT_EXAMINE_COOKBOOK
        }
        actions += list(filter(
            lambda a: a in other_valid_commands, admissible_actions))
        actions += list(filter(
            lambda a: a.startswith("go"), admissible_actions))
        actions += list(filter(
            lambda a: (a.startswith("drop") and
                       all(map(lambda t: t not in a, self._theme_words))),
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
        if not self.is_training:
            if ((self.__winning_recorder[self.game_id] is not None) and
                    (not self.__winning_recorder[self.game_id])):
                for a2remove in self.__actions_to_remove[self.game_id]:
                    try:
                        actions.remove(a2remove)
                        self.debug(
                            "action {} is removed".format(
                                a2remove))
                    except ValueError as _:
                        self.debug(
                            "action {} is not found when remove".format(
                                a2remove))
            else:
                pass
        return actions

    def go_with_floor_plan(self, actions, room):
        local_map = self.floor_plan.get_map(room)
        return (["{} to {}".format(a, local_map.get(a))
                 if a in local_map else a for a in actions])

    def rule_based_policy(self, actions, all_actions, instant_reward):
        # use hard set actions in the beginning and the end of one episode
        if ((not self.is_training) and
                (self.__winning_recorder[self.game_id] is not None) and
                self.__winning_recorder[self.game_id]):
            action = self.__action_recorder[self.game_id][self.in_game_t]
        elif "meal" in self._inventory:
            action = ACT_EAT_MEAL
        elif ACT_EXAMINE_COOKBOOK in actions and not self.__see_cookbook:
            action = ACT_EXAMINE_COOKBOOK
            self.__see_cookbook = True
        elif (self._last_action_desc is not None and
              self._last_action_desc.action == ACT_EXAMINE_COOKBOOK and
              instant_reward <= 0):
            action = ACT_INVENTORY
        else:
            action = None

        if action is not None:
            if not action in all_actions:
                self.debug("eat meal not in action list, adding it in ...")
                self.action_collector.extend([action])
                all_actions = self.action_collector.get_actions()
            action_idx = all_actions.index(action)
        else:
            action_idx = None
        action_desc = ActionDesc(
            action_type=ACT_TYPE_RULE, action_idx=action_idx, action=action)
        return action_desc

    def jitter_go_action(
            self, prev_action_desc, actions, all_actions):
        action_desc = None
        admissible_go_actions = list(
            filter(lambda a: a.startswith("go"), actions))
        if (self.hp.jitter_go and (prev_action_desc.action_type == ACT_TYPE_NN)
                and (prev_action_desc.action in admissible_go_actions)):
            if ((self.is_training and
                 random.random() > 1 - self.hp.jitter_train_prob)
                    or ((not self.is_training) and
                        random.random() > 1 - self.hp.jitter_eval_prob)):
                jitter_go_action = random.choice(admissible_go_actions)
                action_idx = all_actions.index(jitter_go_action)
                action = jitter_go_action
                action_desc = ActionDesc(
                    action_type=ACT_TYPE_JITTER, action_idx=action_idx,
                    action=action)
            else:
                pass
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
                _, action = get_random_1Daction(admissible_actions)
                action_idx = all_actions.index(action)
            else:
                pass
        else:
            pass
        action_desc = ActionDesc(
            action_type=ACT_TYPE_RND_WALK, action_idx=action_idx,
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
        return action_desc

    def get_instant_reward(self, score, master, is_terminal, has_won):
        instant_reward = self.clip_reward(score - self._cumulative_score - 0.1)
        self._cumulative_score = score
        if (master == self._prev_master and
            self._last_action_desc is not None and
            self._last_action_desc.action == self._prev_last_action and
                instant_reward < 0):
            instant_reward = max(
                -1.0, instant_reward + self._cumulative_penalty)
            # self.debug("repeated bad try, decrease reward by {},"
            #            " reward changed to {}".format(
            #     self.prev_cumulative_penalty, instant_reward))
            self._cumulative_penalty = self._cumulative_penalty - 0.1
        else:
            self._prev_last_action = (
                self._last_action_desc.action
                if self._last_action_desc is not None else None)
            self._prev_master = master
            self._cumulative_penalty = -0.1

        # only penalize the final score if the agent choose a bad action.
        # do not penalize if failed because of out-of-steps.
        if is_terminal and not has_won and "you lost" in master:
            self.info("game terminate and fail, final reward change"
                      " from {} to -1".format(instant_reward))
            instant_reward = -1
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
                self._last_action_desc is not None and
                self._last_action_desc.action in self.inv_direction):
            self.floor_plan.extend(
                [(prev_place, self._last_action_desc.action, curr_place),
                 (curr_place, self.inv_direction[self._last_action_desc.action],
                  prev_place)])
        return curr_place

    def train_one_batch(self):
        """
        Train one batch of samples.
        Load target model if not exist, save current model when necessary.
        """
        if self.total_t == self.hp.observation_t:
            self.epoch_start_t = ctime()
        # prepare target nn
        if self.target_model is None:
            if self.hp.delay_target_network != 0:
                # save model for loading of target net
                self.debug("save nn for target net")
                self.saver.save(
                    self.sess, self.chkp_prefix,
                    global_step=tf.train.get_or_create_global_step(
                        graph=self.model.graph))
                self.debug("create and load target net")
                self.target_sess, _, self.target_saver, self.target_model =\
                    self.create_n_load_model(is_training=False)
            else:
                self.debug("target net is the same with nn")
                self.target_model = self.model
                self.target_sess = self.sess
                self.target_saver = self.saver

        self.eps = self.annealing_eps(
            self.hp.init_eps, self.hp.final_eps,
            self.total_t - self.hp.observation_t, self.hp.annealing_eps_t)
        self.train_impl(self.sess, self.total_t,
                        self.train_summary_writer, self.target_sess)

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
            restore_from = tf.train.latest_checkpoint(
                os.path.join(self.model_dir, 'last_weights'))
            self.target_saver.restore(self.target_sess, restore_from)
            self.info("target net load from: {}".format(restore_from))
            self.snapshot_saved = True
            self.info("snapshot saved, ready for evaluation")
            self.epoch_start_t = ctime()

    def feed_memory(
            self, instant_reward, is_terminal, action_mask, next_action_mask):
        original_data = self.memo.append(DRRNMemo(
            tid=self.tjs.get_current_tid(), sid=self.tjs.get_last_sid(),
            gid=self.game_id, aid=self._last_action_desc.action_idx,
            reward=instant_reward, is_terminal=is_terminal,
            action_mask=action_mask, next_action_mask=next_action_mask
        ))
        if isinstance(original_data, DRRNMemo):
            if original_data.is_terminal:
                self.debug("tjs delete {}".format(original_data.tid))
                self.tjs.request_delete_key(original_data.tid)

    def remove_logo(self, first_master):
        lines = first_master.split("\n")
        start_line = 0
        room_regex = "^\s*-= (.*) =-.*"
        for i, l in enumerate(lines):
            room_search = re.search(room_regex, l)
            if room_search is not None:
                start_line = i
                break
            else:
                pass
        modified_master = "\n".join(lines[start_line:])
        self.debug("after logo removing: {}".format(modified_master))
        return modified_master

    def is_negative(self, cleaned_obj):
        negative_stems = ["n't", "not"]
        return any(map(lambda nt: nt in cleaned_obj.split(), negative_stems))

    def update_inventory(self, action):
        verb = action.split()[0]
        noun = " ".join(action.split()[1:])
        if verb == "drop":
            try:
                self._inventory.remove(noun)
            except ValueError as e:
                pass
        elif verb == "take":
            try:
                self._inventory.append(noun)
            except ValueError as e:
                pass
        else:
            raise ValueError("unknown action verb: {}".format(action))

    def collect_new_sample(
            self, obs: List[str], scores: List[int], dones: List[bool],
            infos: Dict[str, List[Any]]):
        master = self.remove_logo(obs[0]) if self.in_game_t == 0 else obs[0]
        if self.hp.apply_dependency_parser:
            cleaned_obs = self.dp.reorder(master)
        else:
            cleaned_obs = self.tokenize(master)

        instant_reward = self.get_instant_reward(
            scores[0], cleaned_obs, dones[0], infos["has_won"][0])

        if self._last_action_desc is not None:
            if self._last_action_desc.action == ACT_EXAMINE_COOKBOOK:
                self.get_theme_words(master)
        if self._last_action_desc is not None:
            if self._last_action_desc.action == ACT_INVENTORY:
                self.get_inventory(master)
        if self._last_action_desc is not None:
            if self._last_action_desc.action.startswith("drop"):
                if not self.is_negative(cleaned_obs):
                    self.update_inventory(self._last_action_desc.action)
        if self._last_action_desc is not None:
            if self._last_action_desc.action.startswith("take"):
                if (not self.is_negative(cleaned_obs)) and ("too many things" not in cleaned_obs):
                    self.update_inventory(self._last_action_desc.action)
        if self._last_action_desc is not None:
            if self._last_action_desc.action.startswith("open"):
                if not self.is_negative(cleaned_obs) and "already open" not in cleaned_obs:
                    self.__obs += " " + cleaned_obs
        if self._last_action_desc is not None:
            if self._last_action_desc.action == ACT_LOOK:
                self.__obs = cleaned_obs

        self.debug("theme words: {}".format(self._theme_words))
        self.debug("inventory: {}".format(self._inventory))

        # use see cookbook again if gain one reward
        if instant_reward > 0:
            self.__see_cookbook = False

        if self.tjs.get_last_sid() > 0:  # pass the 1st master
            self.debug("mode: {}, t: {}, in_game_t: {}, eps: {}, {},"
                       " master: {}, reward: {}, is_terminal: {}".format(
                "train" if self.is_training else "eval", self.total_t,
                self.in_game_t, self.eps, self._last_action_desc,
                cleaned_obs, instant_reward, dones[0]))
        else:
            self.info(
                "mode: {}, master: {}, max_score: {}".format(
                    "train" if self.is_training else "eval", cleaned_obs,
                    infos["max_score"]))

        if self.hp.collect_floor_plan:
            curr_place = self.collect_floor_plan(master, self._prev_place)
            # if place changed, then master should be changed
            if curr_place != self._prev_place:
                self.__obs = cleaned_obs
            self._prev_place = curr_place
        else:
            curr_place = None


        self.debug("obs is {}".format(self.__obs))

        obs_idx = self.index_string(cleaned_obs.split())
        self.tjs.append_master_txt(obs_idx)

        if self.is_training:
            self.debug("use admissible actions")
            actions = [a.lower() for a in infos["admissible_commands"][0]]
        else:
            self.debug("generate admissible actions")
            obs = self.__obs
            inventory = self._inventory
            theme_words = self._theme_words if self._theme_words is not None else []
            actions = self.get_admissible_actions(obs, inventory, theme_words)
        if self.hp.use_original_actions:
            pass
        else:
            actions = self.filter_admissible_actions(actions, curr_place)
        actions = self.go_with_floor_plan(actions, curr_place)
        self.info("admissible actions: {}".format(", ".join(sorted(actions))))
        actions_mask = self.action_collector.extend(actions)
        all_actions = self.action_collector.get_actions()

        if self.tjs.get_last_sid() > 0:  # pass the 1st master
            self.feed_memory(
                instant_reward, dones[0],
                self._last_actions_mask, actions_mask)
        else:
            pass

        return actions, all_actions, actions_mask, instant_reward

    def next_step_action(
            self, actions, all_actions, actions_mask, instant_reward):
        self._last_action_desc = self.choose_action(
            actions, all_actions, actions_mask, instant_reward)
        action = self._last_action_desc.action
        action_idx = self._last_action_desc.action_idx

        self.__per_game_recorder.append(action)

        if self._last_action_desc.action_type == ACT_TYPE_NN:
            self._cnt_action[action_idx] += 0.1
            self.debug(self._cnt_action)
        else:
            self.debug("cnt action ignore hard_set_action")

        self.tjs.append_player_txt(
            self.action_collector.get_action_matrix()[action_idx])

        self._last_actions_mask = actions_mask
        return action

    def get_possible_closed_doors(self, obs):
        doors = re.findall(r'a closed ([a-z \-]+ door)', obs)
        return doors

    def retrieve_item_from_inventory(self, inventory, item):
        for i in inventory:
            if item in i:
                return i
        return None

    def get_admissible_actions(self, obs, inventory, theme_words):
        all_actions = [ACT_PREPARE_MEAL, ACT_LOOK, ACT_INVENTORY]
        inventory_sent = " ".join(inventory)

        if "cookbook" in obs:
            all_actions += [ACT_EXAMINE_COOKBOOK]

        for d in ["north", "south", "east", "west"]:
            if d in obs.split():
                all_actions += ["go {}".format(d)]

        if "fridge" in obs:
            all_actions += ["open fridge"]
        if "door" in obs:
            doors = self.get_possible_closed_doors(obs)
            for d in doors:
                all_actions += ["open {}".format(d)]

        cookwares = ["stove", "oven", "bbq"]
        knife_usage = ["dice", "slice", "chop"]
        for c in cookwares:
            if c in obs:
                for t in theme_words:
                    if t in inventory_sent:
                        t_with_status = self.retrieve_item_from_inventory(inventory, t)
                        if t_with_status is None:
                            t_with_status = t
                        all_actions += ["cook {} with {}".format(t_with_status, c)]
        if "knife" in inventory_sent:
            for t in theme_words:
                if t in inventory_sent:
                    t_with_status = self.retrieve_item_from_inventory(
                        inventory, t)
                    if t_with_status is None:
                        t_with_status = t
                    all_actions += (
                        ["{} {} with knife".format(k, t_with_status)
                         for k in knife_usage])
        if "knife" in obs:
            all_actions += ["take knife"]
        for t in theme_words:
            if t in obs and t not in inventory:
                all_actions += ["take {}".format(t)]

        # active drop actions only after we know the theme words
        if len(theme_words) != 0:
            all_actions += ["drop {}".format(i) for i in inventory]

        if "meal" in inventory_sent:
            all_actions += [ACT_EAT_MEAL]

        return all_actions


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

        (actions, all_actions, actions_mask, instant_reward
         ) = self.collect_new_sample(obs, scores, dones, infos)

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
        # revert back go actions for the game playing
        if player_t.startswith("go"):
            player_t = " ".join(player_t.split()[:2])
        return [player_t] * len(obs)
