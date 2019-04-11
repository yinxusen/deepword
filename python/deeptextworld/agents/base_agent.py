import glob
import hashlib
import os
import random
import re
import string
from typing import List, Dict, Any, Optional

import collections
import numpy as np
import tensorflow as tf
from bitarray import bitarray
from nltk import word_tokenize

from deeptextworld import trajectory
from deeptextworld.action import ActionCollector
from deeptextworld.floor_plan import FloorPlanCollector
from deeptextworld.hparams import save_hparams, output_hparams, copy_hparams
from deeptextworld.log import Logging
from deeptextworld.tree_memory import TreeMemory
from deeptextworld.utils import get_token2idx, load_vocab, load_actions, ctime


class DRRNMemo(collections.namedtuple(
    "DRRNMemo",
    ("tid", "sid", "gid", "aid", "reward", "is_terminal", "action_mask"))):
    pass


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
            "go south": "go north", "go north": "go south",
            "go east": "go west", "go west": "go east"}

        self.hp, self.tokens, self.token2idx = self.init_tokens(hp)
        self.info(output_hparams(self.hp))

        self.tjs = None
        self.memo = None
        self.model = None
        self.target_model = None
        self.action_collector = None
        self.floor_plan = None

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
        self._last_action_idx = None
        self._last_actions_mask = None
        self._last_action = None
        self.game_id = None
        self.prev_report = None

        self.cumulative_score = 0
        self.snapshot_saved = False
        self.epoch_start_t = 0
        self.prev_cumulative_penalty = -0.1
        self.prev_player_t = None
        self.prev_master_t = None

        self.empty_trans_table = str.maketrans("", "", string.punctuation)
        self.theme_words = None
        self.see_cookbook = False
        self.prev_place = None


    @classmethod
    def init_tokens(cls, hp):
        """
        :param hp:
        :return:
        """
        new_hp = copy_hparams(hp)
        # make sure that padding_val is indexed as 0.
        additional_tokens = [hp.padding_val, hp.unk_val, hp.sos, hp.eos]
        tokens = additional_tokens + list(load_vocab(hp.vocab_file))
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

    @classmethod
    def lower_tokenize(cls, master):
        return ' '.join([t.lower() for t in word_tokenize(master)])

    def preprocess_master_output(self, master):
        return ' '.join(
            map(lambda t: t.lower(),
                filter(lambda t: t.isalpha(),
                       word_tokenize(
                           master.translate(self.empty_trans_table)))))

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
                self.info("Try to restore parameters from: {}".format(restore_from))
                saver.restore(train_sess, restore_from)
                if not self.hp.start_t_ignore_model_t:
                    global_step = tf.train.get_or_create_global_step()
                    trained_steps = train_sess.run(global_step)
                    start_t = trained_steps + self.hp.observation_t
            else:
                self.info('No checkpoint to load, training from scratch')
            trainable_vars = tf.trainable_variables()
            self.info('trainable variables: {}'.format(trainable_vars))
            self.info('count of trainable vars w/o src_embeddings: {}'.format(
                self.count_trainable(trainable_vars, mask='src_embeddings')))
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
            self.eps = 0.05
            self.total_t = 0
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
        recipe = infos["extra.recipe"]
        # use stronger game identity
        self.game_id = hashlib.md5(
            (obs[0] + recipe[0]).encode("utf-8")).hexdigest()
        self.action_collector.add_new_episode(eid=self.game_id)
        self.floor_plan.add_new_episode(eid=self.game_id)
        self.in_game_t = 0
        self.cumulative_score = 0
        self._episode_has_started = True
        self.prev_place = None

        theme_regex = ".*Ingredients:<\|>(.*)<\|>Directions.*"
        theme_words_search = re.search(theme_regex, recipe[0].replace("\n", "<|>"))
        self.see_cookbook = False
        self.theme_words = None
        if theme_words_search:
            theme_words = theme_words_search.group(1)
            self.theme_words = list(
                filter(lambda w: w != "",
                       map(lambda w: w.strip(), theme_words.split("<|>"))))
            self.debug("theme words: {}".format(", ".join(self.theme_words)))

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
        self._episode_has_started = False
        self._last_action_idx = None
        self._last_actions_mask = None
        self._last_action = None
        self.game_id = None

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
        if self.theme_words is None:
            self.debug("no theme word found, use all actions")
            return actions, []
        contained = []
        others = []
        for a in actions:
            if self.contain_words(a, self.theme_words):
                contained.append(a)
            else:
                others.append(a)

        return contained, others

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
        immediate_reward = self.clip_reward(scores[0] - self.cumulative_score - 0.1)
        self.cumulative_score = scores[0]

        if self.in_game_t == 0:
            # to avoid adding the text world logo in the first sentence
            master = infos["description"][0]
        else:
            master = obs[0]
        cleaned_obs = self.lower_tokenize(master)

        curr_place = None
        if self.hp.collect_floor_plan:
            room_regex = ".*-= (.*) =-.*"
            room_search = re.search(room_regex, cleaned_obs)
            if room_search is not None:
                curr_place = room_search.group(1)
                if self.prev_place is None:
                    self.prev_place = curr_place
            else:
                self.debug("no match place from {}".format(cleaned_obs))
                curr_place = self.prev_place
            curr_map = self.floor_plan.get_map(curr_place)
            cleaned_obs = cleaned_obs + " " + curr_map

        if (cleaned_obs == self.prev_master_t and
            self._last_action == self.prev_player_t and immediate_reward < 0):
            immediate_reward = max(-1.0, immediate_reward + self.prev_cumulative_penalty)
            self.debug("repeated bad try, decrease reward by {}, reward changed to {}".format(
                self.prev_cumulative_penalty, immediate_reward))
            self.prev_cumulative_penalty = self.prev_cumulative_penalty - 0.1
        else:
            self.prev_player_t = self._last_action
            self.prev_master_t = cleaned_obs
            self.prev_cumulative_penalty = -0.1

        if dones[0] and not infos["has_won"][0]:
            self.info("game terminate and fail,"
                      " final reward change from {} to -1".format(immediate_reward))
            immediate_reward = -1

        obs_idx = self.index_string(cleaned_obs.split())
        self.tjs.append_master_txt(obs_idx)

        if self.tjs.get_last_sid() > 0:  # pass the 1st master
            self.debug("mode: {}, t: {}, in_game_t: {}, eps: {}, {},"
                       " master: {}, reward: {}, is_terminal: {}".format(
                "train" if self.is_training else "eval", self.total_t,
                self.in_game_t, self.eps, self.report_status(self.prev_report),
                cleaned_obs, immediate_reward, dones[0]))

            tid_ = self.tjs.get_current_tid()
            sid_ = self.tjs.get_last_sid()
            # importance_to_repeat = 10 if immediate_reward > 0 else 1
            # if importance_to_repeat > 1:
            #     self.info("encounter important sample, repeat 10 times in memo")
            importance_to_repeat = 1 if immediate_reward > 0 else 1
            for _ in range(importance_to_repeat):
                self.memo.append(DRRNMemo(
                    tid=tid_, sid=sid_,
                    gid=self.game_id, aid=self._last_action_idx,
                    reward=immediate_reward,
                    is_terminal=dones[0], action_mask=self._last_actions_mask))
            if (self.hp.collect_floor_plan and curr_place != self.prev_place
                    and self._last_action.startswith("go")):
                self.floor_plan.extend(
                    [(self.prev_place, self._last_action, curr_place),
                     (curr_place, self.inv_direction[self._last_action], self.prev_place)])
                self.prev_place = curr_place
        else:
            self.info("mode: {}, master: {}, max_score: {}".format(
                "train" if self.is_training else "eval", cleaned_obs,
                infos["max_score"]))

        # notice the position of all(dones)
        # make sure add the last action-master pair into memory
        if all(dones):
            self._end_episode(obs, scores, infos)
            return  # Nothing to return.

        # populate my own admissible actions
        admissible_commands = infos["admissible_commands"][0]
        contained, others = self.contain_theme_words(admissible_commands)
        actions = ["inventory", "look"]
        actions += contained
        admissible_go = list(filter(lambda a: a.startswith("go"), admissible_commands))
        actions += admissible_go
        actions = list(filter(lambda c: not c.startswith("examine"), actions))
        actions = list(filter(lambda c: not c.startswith("close"), actions))
        actions = list(filter(lambda c: not c.startswith("insert"), actions))
        actions = list(filter(lambda c: not c.startswith("eat"), actions))
        # drop needs
        # actions = list(filter(lambda c: not c.startswith("drop"), actions))
        actions = list(filter(lambda c: not c.startswith("put"), actions))
        other_valid_commands = {"prepare meal", "eat meal", "examine cookbook"}
        actions += list(filter(lambda a: a in other_valid_commands, admissible_commands))
        actions += list(filter(
            lambda a: (a.startswith("drop") and
                       all(map(lambda t: t not in a, self.theme_words))), others))
        actions += list(filter(lambda a: a.startswith("take") and "knife" in a, others))
        actions += list(filter(lambda a: a.startswith("open"), others))
        self.debug("previous admissible actions: {}".format(", ".join(sorted(admissible_commands))))
        self.debug("new admissible actions: {}".format(", ".join(sorted(actions))))

        actions_mask = self.action_collector.extend(actions)
        all_actions = self.action_collector.get_actions()

        # use hard set actions in the beginning and the end of one episode
        if "examine cookbook" in actions and not self.see_cookbook:
            player_t = "examine cookbook"
            action_idx = all_actions.index(player_t)
            self.prev_report = [('hard_set_action', action_idx),
                                ('action', player_t)]
            self.see_cookbook = True
        elif self._last_action == "examine cookbook":
            player_t = "inventory"
            action_idx = all_actions.index(player_t)
            self.prev_report = [('hard_set_action', action_idx),
                                ('action', player_t)]
        elif self._last_action == "prepare meal" and immediate_reward > 0:
            player_t = "eat meal"
            if not player_t in all_actions:
                self.debug("eat meal not in action list, adding it in ...")
                self.action_collector.extend([player_t])
                all_actions = self.action_collector.get_actions()
            action_idx = all_actions.index(player_t)
            self.prev_report = [('hard_set_action', action_idx),
                                ('action', player_t)]
        else:
            action_idx, player_t, self.prev_report= self.get_an_eps_action(actions_mask)

            # add jitter to go actions to avoid overfitting
            if (self.hp.jitter_go and (self.prev_report[0][0] == "action")
                    and (player_t in admissible_go)):
                if ((self.is_training and random.random() > 1 - self.hp.jitter_train_prob)
                        or ((not self.is_training) and random.random() > 1 - self.hp.jitter_eval_prob)):
                    original_action = player_t
                    jitter_go_action = random.choice(admissible_go)
                    action_idx = all_actions.index(jitter_go_action)
                    player_t = jitter_go_action
                    self.prev_report = ([("original action", original_action),
                                         ("jitter go action", player_t),
                                         ("# admissible go", len(admissible_go))])
                else:
                    pass

        self.tjs.append_player_txt(
            self.action_collector.get_action_matrix()[action_idx])
        self._last_action_idx = action_idx
        self._last_actions_mask = actions_mask
        self._last_action = player_t
        if self.is_training and self.total_t >= self.hp.observation_t:
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
        self.total_t += 1
        self.in_game_t += 1
        return [player_t] * len(obs)
