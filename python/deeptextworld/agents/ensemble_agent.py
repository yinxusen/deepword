"""
Ensemble agent only works for evaluation
"""
import os
import random
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf

from deeptextworld.agents.drrn_agent import DRRNAgent
from deeptextworld.dqn_func import get_best_1Daction
from deeptextworld.dqn_func import get_random_1Daction


class EnsembleAgent(DRRNAgent):
    def __init__(self, hp, model_dir):
        super(EnsembleAgent, self).__init__(hp, model_dir)
        self.is_training = False
        self.models = None

    def create_n_load_eval_model(self, ckpt_path):
        model = self.create_eval_model_instance()
        self.info("create eval model")
        eval_conf = tf.ConfigProto(log_device_placement=False,
                                    allow_soft_placement=True)
        eval_sess = tf.Session(graph=model.graph, config=eval_conf)
        with model.graph.as_default():
            eval_sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=self.hp.max_snapshot_to_keep,
                                   save_relative_paths=True)
            restore_from = tf.train.latest_checkpoint(ckpt_path)
            if restore_from is not None:
                # Reload weights from directory if specified
                self.info("Try to restore parameters from: {}".format(restore_from))
                saver.restore(eval_sess, restore_from)
            else:
                return None
        return eval_sess, model

    def eval(self, load_best=True):
        ckpt_paths = [os.path.join(self.model_dir, "tier1/best_weights"),
                      os.path.join(self.model_dir, "tier2/best_weights"),
                      os.path.join(self.model_dir, "tier3/best_weights"),
                      os.path.join(self.model_dir, "tier4/best_weights"),
                      os.path.join(self.model_dir, "tier5/best_weights"),
                      os.path.join(self.model_dir, "tier6/best_weights")]
        self.models = list(
            filter(lambda m: m is not None,
                   map(lambda p: self.create_n_load_eval_model(p), ckpt_paths)))

    def train(self):
        raise NotImplementedError()

    def act(self, obs: List[str], scores: List[int], dones: List[bool],
            infos: Dict[str, List[Any]]):
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
        actions = list(filter(lambda c: not c.startswith("drop"), actions))
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
            action_mask = self.fromBytes([actions_mask])[0]
            if np.random.random() < self.eps:
                action_idx, player_t = get_random_1Daction(
                    self.action_collector.get_actions(), action_mask)
                self.prev_report = [('random_action', action_idx),
                                    ('action', player_t)]
            else:
                action_matrix = self.action_collector.get_action_matrix()
                actions_len = self.action_collector.get_action_len()
                indexed_state_t, lens_t = self.tjs.fetch_last_state()

                q_actions_t = np.zeros(self.hp.n_actions)
                for sess, model in self.models:
                    q_vec = sess.run(model.q_actions, feed_dict={
                        model.src_: [indexed_state_t],
                        model.src_len_: [lens_t],
                        model.actions_mask_: [action_mask],
                        model.actions_: [action_matrix],
                        model.actions_len_: [actions_len]
                    })[0]
                    q_actions_t = q_actions_t + q_vec

                action_idx, q_max, player_t = get_best_1Daction(
                    q_actions_t, self.action_collector.get_actions(),
                    mask=action_mask)
                self.prev_report = [('action', player_t), ('q_max', q_max),
                                    ('q_argmax', action_idx)]

            # add jitter to go actions to avoid overfitting
            if (self.hp.jitter_go and (self.prev_report[0][0] == "action")
                    and (player_t in admissible_go)):
                if (self.is_training and random.random() > 0.5) or (not self.is_training):
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
        self.total_t += 1
        self.in_game_t += 1
        return [player_t] * len(obs)

