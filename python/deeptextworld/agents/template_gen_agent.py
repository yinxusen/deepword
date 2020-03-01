from abc import ABC

from deeptextworld.agents.base_agent import *


class TemplateGenAgent(BaseAgent, ABC):
    def __init__(self, hp, model_dir):
        super(TemplateGenAgent, self).__init__(hp, model_dir)
        self._require_drop_actions = False
        self._inventory = None
        self._see_inventory = False
        self._obs = None
        self._connections = {}

    @classmethod
    def filter_contradicted_actions(cls, actions):
        """
        TODO: useless for now
        :param actions:
        :return:
        """
        contradicted = [False] * len(actions)
        for i in range(len(actions)):
            tokens = actions[i].split()
            if "dice" in tokens and "diced" in tokens:
                contradicted[i] = True
            if "chop" in tokens and "chopped" in tokens:
                contradicted[i] = True
            if "slice" in tokens and "sliced" in tokens:
                contradicted[i] = True
            if "bbq" in tokens and "grilled" in tokens:
                contradicted[i] = True
            if "stove" in tokens and "fried" in tokens:
                contradicted[i] = True
            if "oven" in tokens and "roasted" in tokens:
                contradicted[i] = True
        valid_actions = filter(
            lambda a_c: not a_c[1], zip(actions, contradicted))
        return list(map(lambda a_c: a_c[0], valid_actions))

    @classmethod
    def get_possible_closed_doors(cls, obs):
        doors = re.findall(r"a closed ([a-z \-]+ door)", obs)
        return doors

    @classmethod
    def retrieve_name_from_inventory(cls, inventory, item):
        for thing in inventory:
            if item in thing:
                return thing
        return None

    @classmethod
    def update_inventory(cls, action, inventory_list):
        action_obj = " ".join(action.split()[1:])
        if action.startswith("drop"):
            new_inventory_list = list(filter(
                lambda x: x != action_obj, inventory_list))
        elif action.startswith("take"):
            new_inventory_list = inventory_list + [action_obj]
        else:
            raise ValueError("unknown action verb: {}".format(action))
        return new_inventory_list

    @classmethod
    def remove_logo(cls, first_master):
        lines = first_master.split("\n")
        start_line = 0
        room_regex = r"^\s*-= (.*) =-.*"
        for i, l in enumerate(lines):
            room_search = re.search(room_regex, l)
            if room_search is not None:
                start_line = i
                break
            else:
                pass
        modified_master = "\n".join(lines[start_line:])
        return modified_master

    @classmethod
    def is_negative(cls, cleaned_obj):
        negative_stems = ["n't", "not"]
        return any(map(lambda nt: nt in cleaned_obj.split(), negative_stems))

    @classmethod
    def is_observation(cls, master):
        """if there is room name then it's an observation other a dialogue"""
        return cls.get_room_name(master) is not None

    @classmethod
    def get_connections(cls, raw_recipe, theme_words):
        connections = {}
        lines = list(
            filter(
                lambda x: x != "",
                map(lambda x: x.strip(), raw_recipe.split("\n"))))
        start_line = 0
        directions_regex = r"^\sDirections"
        for i, line in enumerate(lines):
            d_search = re.search(directions_regex, line)
            if d_search is not None:
                start_line = i
                break
            else:
                pass
        lines = lines[start_line:]
        for line in lines:
            for t in theme_words:
                if t in line:
                    if t not in connections:
                        connections[t] = set()
                    connections[t].add(line.split()[0])
                else:
                    pass
        return connections

    @classmethod
    def get_inventory(cls, inventory_list):
        items = list(filter(
            lambda s: len(s) != 0,
            map(lambda s: s.strip(), inventory_list.split("\n"))))
        if len(items) > 1:
            # remove the first a/an/the/some ...
            items = list(map(lambda i: " ".join(i.split()[1:]), items[1:]))
        else:
            items = []
        return items

    @classmethod
    def negative_response_reward(cls, master):
        negative_response_penalty = 0
        if (not cls.is_observation(master)) and cls.is_negative(master):
            negative_response_penalty = 0.5
        return negative_response_penalty

    def select_additional_infos(self):
        """
        additional information needed when playing the game
        requested infos here are required to run the Agent
        """
        return EnvInfos(
            max_score=True,
            won=True)

    def _start_episode_impl(self, obs, infos):
        super(TemplateGenAgent, self)._start_episode_impl(obs, infos)
        if self.game_id not in self._connections:
            self._connections[self.game_id] = {}
        self._require_drop_actions = False
        self._inventory = []
        self._obs = None
        self._see_inventory = False

    def get_admissible_actions(self, infos=None):
        obs = self._obs
        inventory = self._inventory
        theme_words = self._theme_words[self.game_id]
        connections = self._connections[self.game_id]

        all_actions = [ACT.prepare_meal, ACT.look, ACT.inventory]
        inventory_sent = " ".join(inventory)

        if "cookbook" in obs:
            all_actions += [ACT.examine_cookbook]

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
        cook_verbs = ["fry", "roast", "grill"]
        knife_usage = ["dice", "slice", "chop"]
        knife_verbs = ["dice", "slice", "chop"]
        all_possible_verbs = cook_verbs + knife_verbs
        for c, v in zip(cookwares, cook_verbs):
            if c in obs:
                for t in theme_words:
                    if t in inventory_sent:
                        if (t in connections) and (
                                (v in connections[t]) or
                                all(map(lambda x: x not in all_possible_verbs,
                                        connections[t]))):
                            t_with_status = self.retrieve_name_from_inventory(
                                inventory, t)
                            if t_with_status is None:
                                t_with_status = t
                            all_actions += (
                                ["cook {} with {}".format(t_with_status, c)])
                        else:
                            pass
        if "knife" in inventory_sent:
            for t in theme_words:
                if t in inventory_sent:
                    t_with_status = self.retrieve_name_from_inventory(
                        inventory, t)
                    if t_with_status is None:
                        t_with_status = t
                    for k, v in zip(knife_usage, knife_verbs):
                        if (t in connections) and (
                                (v in connections[t]) or
                                all(map(lambda x: x not in all_possible_verbs,
                                        connections[t]))):
                            all_actions += (
                                ["{} {} with knife".format(k, t_with_status)])
                        else:
                            pass
        if "knife" in obs:
            all_actions += ["take knife"]

        for t in theme_words:
            if t in obs and t not in inventory:
                all_actions += ["take {}".format(t)]

        # active drop actions only after we know the theme words
        drop_actions = []
        if len(theme_words) != 0:
            for i in inventory:
                if all(map(lambda tw: tw not in i, theme_words)):
                    drop_actions += ["drop {}".format(i)]
            # drop useless items first
            # if there is no useless items, drop useful ingredients
            if self._require_drop_actions and (len(drop_actions) == 0):
                drop_actions += ["drop {}".format(i) for i in inventory]
            all_actions += drop_actions

        if "meal" in inventory_sent:
            all_actions += [ACT.eat_meal]

        return all_actions

    def _get_master_starter(self, obs, infos):
        return self.remove_logo(obs[0])

    def update_status_impl(self, master, cleaned_obs, instant_reward, infos):
        """
        Update game status according to observations and instant rewards.
        the following vars are updated:
          1. theme words
          2. connections
          3. inventory
          4. observation
          5. whether requires drop actions or not

        :param master:
        :param cleaned_obs:
        :param instant_reward:
        :param infos:
        :return:
        """
        super(TemplateGenAgent, self).update_status_impl(
            master, cleaned_obs, instant_reward, infos)
        if self._curr_place != self._prev_place:
            self._obs = cleaned_obs
        if self._last_action is not None:
            if self._last_action.action == ACT.examine_cookbook:
                self._theme_words[self.game_id] = self.get_theme_words(
                    master)
                self._connections[self.game_id] = self.get_connections(
                    master, self._theme_words[self.game_id])
            elif self._last_action.action == ACT.inventory:
                self._inventory = self.get_inventory(master)
            elif self._last_action.action.startswith("drop"):
                if not self.is_negative(cleaned_obs):
                    self._inventory = self.update_inventory(
                        self._last_action.action, self._inventory)
                    self._require_drop_actions = False
            elif self._last_action.action.startswith("take"):
                if ((not self.is_negative(cleaned_obs)) and
                        ("too many things" not in cleaned_obs)):
                    self._inventory = self.update_inventory(
                        self._last_action.action, self._inventory)
                    self._require_drop_actions = False
                if "too many things" in cleaned_obs:
                    self._require_drop_actions = True
            elif self._last_action.action.startswith("open"):
                if ((not self.is_negative(cleaned_obs)) and
                        ("already open" not in cleaned_obs)):
                    self._obs += " " + cleaned_obs
            elif self._last_action.action == ACT.look:
                self._obs = cleaned_obs
            else:
                pass
        else:
            self.debug(
                "last action description is None, nothing to update")
        self.debug("theme words: {}".format(self._theme_words[self.game_id]))
        self.debug("inventory: {}".format(self._inventory))
        self.debug("obs: {}".format(self._obs))

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
        if ACT.inventory in actions and not self._see_inventory:
            action = ACT.inventory
            self._see_inventory = True
        elif "meal" in self._inventory:
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
