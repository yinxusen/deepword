import random
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

    def rule_based_policy(self, actions, all_actions, instant_reward):
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
            action=action, q_actions=None)
        return action_desc

    def collect_new_sample(self, master, instant_reward, dones, infos):
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
        admissible_actions = self.filter_admissible_actions(admissible_actions)
        effective_actions = self.go_with_floor_plan(admissible_actions)
        action_mask = self.actor.extend(effective_actions)
        all_actions = self.actor.actions
        # TODO: use all actions instead of using admissible actions
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

    def choose_action(
            self, actions, all_actions, action_mask, instant_reward):
        # when q_actions is required to get, this should be True
        if self.hp.compute_policy_action_every_step:
            policy_action_desc = self.get_policy_action(action_mask)
        else:
            policy_action_desc = None

        action_desc = self.rule_based_policy(
            actions, all_actions, instant_reward)
        if action_desc.action_idx is None:
            action_desc = self.random_walk_for_collecting_fp(
                actions, all_actions)
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

    def _start_episode_impl(self, obs, infos):
        super(CompetitionAgent, self)._start_episode_impl(obs, infos)
        if self.game_id not in self._theme_words:
            self._theme_words[self.game_id] = []
        self._see_cookbook = False

    def update_status(self, obs, scores, dones, infos):
        self._prev_place = self._curr_place
        master = infos[INFO_KEY.desc][0] if self.in_game_t == 0 else obs[0]

        if (not self._theme_words[self.game_id]
                and self._last_action is not None
                and self._last_action.action == ACT.examine_cookbook):
            self._theme_words[self.game_id] = self.get_theme_words(master)
            self.debug(
                "get theme words: {}".format(self._theme_words[self.game_id]))

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
