import re

from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.agents.utils import *


class CompetitionAgent(BaseAgent):
    def __init__(self, hp, model_dir):
        super(CompetitionAgent, self).__init__(hp, model_dir)
        self._theme_words = {}
        self._see_cookbook = False

    @classmethod
    def contain_words(cls, sentence, words):
        return any(map(lambda w: w in sentence, words))

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

    def filter_admissible_actions(self, admissible_actions):
        """
        Filter unnecessary actions.
        TODO: current filtering logic has problem. e.g. when there is no
          theme words, then all actions are moved in contained.
          Most of them will be filtered out because of this, which is wrong.
          Fortunately for cooking game it's OK for now.
          Do not apply this action filter for other games.
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
        actions = list(set(actions))
        return actions

    def rule_based_policy(self, actions, instant_reward):
        # TODO: use see cookbook again if gain one reward
        if instant_reward > 0:
            self._see_cookbook = False

        if (self._last_action is not None
                and self._last_action.action == ACT.prepare_meal
                and instant_reward > 0):
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

        if action is not None and action in actions:
            action_idx = self.actor.action2idx.get(action)
            return ActionDesc(
                action_type=ACT_TYPE.rule, action_idx=action_idx,
                token_idx=self.actor.action_matrix[action_idx],
                action_len=self.actor.action_len[action_idx],
                action=action, q_actions=None)
        else:
            return None

    def prepare_actions(self, admissible_actions: List[str]) -> List[str]:
        actions = super(CompetitionAgent, self).prepare_actions(
            admissible_actions)
        actions = self.filter_admissible_actions(actions)
        return actions

    def _start_episode_impl(self, obs, infos):
        super(CompetitionAgent, self)._start_episode_impl(obs, infos)
        if self.game_id not in self._theme_words:
            self._theme_words[self.game_id] = []
        self._see_cookbook = False

    def update_status(self, obs, scores, dones, infos):
        master, instant_reward = super(CompetitionAgent, self).update_status(
            obs, scores, dones, infos)

        if (not self._theme_words[self.game_id]
                and self._last_action is not None
                and self._last_action.action == ACT.examine_cookbook):
            self._theme_words[self.game_id] = self.get_theme_words(master)
            self.debug(
                "get theme words: {}".format(self._theme_words[self.game_id]))

        return master, instant_reward
