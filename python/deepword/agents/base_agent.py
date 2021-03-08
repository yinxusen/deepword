import random
import re
from os import path
from os import remove
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from scipy.stats import entropy
from tensorflow.contrib.training import HParams
from termcolor import colored
from textworld import EnvInfos

from deepword.action import ActionCollector
from deepword.agents.utils import ACT, ObsInventory, INFO_KEY, \
    ActionDesc, ACT_TYPE, Memolet, ActionMaster, Logging, LinearDecayedEPS
from deepword.agents.utils import categorical_without_replacement, \
    get_best_1d_q, remove_zork_version_info, get_path_tags, get_hash_state
from deepword.floor_plan import FloorPlanCollector
from deepword.hparams import save_hparams, output_hparams, conventions
from deepword.tokenizers import init_tokens, Tokenizer
from deepword.trajectory import Trajectory
from deepword.tree_memory import TreeMemory
from deepword.utils import get_hash, core_name2clazz
from deepword.utils import report_status


class BaseAgent(Logging):
    """
    Base agent class that using
     1. action collector
     2. trajectory collector
     3. floor plan collector
     4. tree memory storage and sampling
    """

    def __init__(self, hp: HParams, model_dir: str) -> None:
        """
        Initialize a base agent

        Args:
            hp: hyper-parameters, refer to :py:mod:`deepword.hparams`
            model_dir: path to model dir
        """
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

        self.hp, self.tokenizer = init_tokens(hp)
        self.info(output_hparams(self.hp))

        core_class = core_name2clazz(self.hp.core_clazz)
        self.core = core_class(self.hp, self.model_dir)

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
        self._positive_scores = 0
        self._negative_scores = 0
        self._objective = ""
        self._objective_ids = []
        self._walkthrough = []
        self._continue_walkthrough = False
        if self.hp.append_objective_to_tj:
            self._loaded_objectives = self.load_objectives(
                conventions.objective_file)
        else:
            self._loaded_objectives = dict()

    @classmethod
    def load_objectives(cls, fn_objectives):
        with open(fn_objectives, "r") as f:
            objectives = [x.strip().split() for x in f.readlines()]
            objectives = dict([(x[0], (x[1], x[2])) for x in objectives])
        return objectives

    @classmethod
    def _clip_reward(cls, reward: float) -> float:
        """clip reward into [-1, 1]"""
        return max(min(reward, 1), -1)

    @classmethod
    def _get_room_name(cls, master: str) -> Optional[str]:
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
    def _walkthrough_prob_per_step(
            cls, n_steps: int, prob_complete_play: float) -> float:
        """
        compute probability of using walkthrough per step

        Args:
            n_steps: number steps of walkthrough
            prob_complete_play: the probability of completing walkthrough

        Returns: probability of using walkthrough per step
        """
        assert n_steps > 0, "walkthrough steps should larger than 0"
        return np.exp((1 / n_steps) * np.log(prob_complete_play))

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
            admissible_commands=True,
            objective=True,
            extras=["walkthrough"])

    def _get_admissible_actions(
            self, infos: Dict[str, List[Any]]) -> List[str]:
        """
        We add inventory and look, in case that the game doesn't provide these
        two key actions.
        """
        # system provided admissible actions
        sys_actions = [a.lower() for a in infos[INFO_KEY.actions][0]]
        admissible_actions = list(set(sys_actions) | {ACT.inventory, ACT.look})
        return admissible_actions

    # TODO: it's possible for different games to have the same game ID
    # TODO: change it to a better method, maybe read game name from infos
    @classmethod
    def _compute_game_id(cls, infos: Dict[str, List[Any]]) -> str:
        assert INFO_KEY.desc in infos, "request description is required"
        assert INFO_KEY.inventory in infos, "request inventory is required"
        starter = "{}\n{}".format(
            infos[INFO_KEY.desc][0], infos[INFO_KEY.inventory][0])
        return get_hash(starter)

    def _init_actions(
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

    def _init_trajectory(
            self, hp: HParams, tjs_path: str, with_loading=True) -> Trajectory:
        tjs = Trajectory(num_turns=hp.num_turns)
        if with_loading:
            try:
                tjs.load_tjs(tjs_path)
            except IOError as e:
                self.info("load trajectory error: \n{}".format(e))
        return tjs

    def _init_state_text(self, state_text_path, with_loading=True):
        # num_turns = 1, we only need the most recent ObsInventory
        stc = Trajectory(num_turns=1)
        if with_loading:
            try:
                stc.load_tjs(state_text_path)
            except IOError as e:
                self.info("load trajectory error: \n{}".format(e))
        return stc

    def _init_memo(
            self, hp: HParams, memo_path: str, with_loading=True) -> TreeMemory:
        memory = TreeMemory(capacity=hp.replay_mem)
        if with_loading:
            try:
                memory.load_memo(memo_path)
            except IOError as e:
                self.info("load memory error: \n{}".format(e))
        return memory

    def _init_floor_plan(
            self, fp_path: str, with_loading=True) -> FloorPlanCollector:
        fp = FloorPlanCollector()
        if with_loading:
            try:
                fp.load_fps(fp_path)
            except IOError as e:
                self.info("load floor plan error: \n{}".format(e))
        return fp

    def _get_a_random_action(self, action_mask: np.ndarray) -> ActionDesc:
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
        """
        self.is_training = True
        self._init()

    def eval(self, load_best=True) -> None:
        """
        call eval() before performing evaluation

        Args:
            load_best: load from the best weights, otherwise from last weights
        """
        self.is_training = False
        self._init(load_best)

    def reset(self, restore_from: Optional[str] = None) -> None:
        """
        reset is only used for evaluation during training
        do not use it at anywhere else.

        Args:
            restore_from: where to restore the model, `None` goes to default
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
        return path.join(
            self.model_dir, "{}-{}.npz".format(prefix, tag))

    def _get_context_obj_path(self, prefix: str) -> str:
        return self._get_context_obj_path_w_tag(prefix, self._largest_valid_tag)

    def _get_context_obj_new_path(self, prefix: str) -> str:
        return self._get_context_obj_path_w_tag(prefix, self.total_t)

    def _load_context_objs(self) -> None:
        valid_tags = self._get_compatible_snapshot_tag()
        self._largest_valid_tag = max(valid_tags) if len(valid_tags) != 0 else 0
        self.info("try to load from tag: {}".format(self._largest_valid_tag))

        action_path = self._get_context_obj_path(self.action_prefix)
        tjs_path = self._get_context_obj_path(self.tjs_prefix)
        memo_path = self._get_context_obj_path(self.memo_prefix)
        fp_path = self._get_context_obj_path(self.fp_prefix)
        stc_path = self._get_context_obj_path(self.stc_prefix)

        # always loading actions to avoid different action index for DQN
        self.actor = self._init_actions(
            self.hp, self.tokenizer, action_path,
            with_loading=self.is_training)
        self.tjs = self._init_trajectory(
            self.hp, tjs_path, with_loading=self.is_training)
        self.memo = self._init_memo(
            self.hp, memo_path, with_loading=self.is_training)
        self.floor_plan = self._init_floor_plan(
            fp_path, with_loading=self.is_training)
        # load stc
        self.stc = self._init_state_text(stc_path, with_loading=True)

    def _init_impl(
            self, load_best: bool = False,
            restore_from: Optional[str] = None) -> None:
        self._load_context_objs()
        self.core.init(
            is_training=self.is_training, load_best=load_best,
            restore_from=restore_from)

        if self.is_training:
            # save hparams if training
            save_hparams(self.hp, path.join(self.model_dir, 'hparams.json'))
            if self.hp.start_t_ignore_model_t:
                self.total_t = min(
                    self.hp.observation_t,
                    len(self.memo) if self.memo is not None else 0)
            else:
                self.total_t = min(
                    self.hp.observation_t + self.core.loaded_ckpt_step,
                    len(self.memo) if self.memo is not None else 0)
        else:
            self.total_t = 0
            if self.hp.policy_to_action.lower() == "Sampling".lower():
                self.eps = 0
            elif self.hp.policy_to_action.lower() == "LinUCB".lower():
                self.eps = self.hp.policy_eps
            elif self.hp.policy_to_action.lower() == "EPS".lower():
                self.eps = self.hp.policy_eps
            else:
                raise ValueError("Unknown policy utilization method: {}".format(
                    self.hp.policy_to_action))

    def _start_episode(
            self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        """
        Prepare the agent for the upcoming episode.
        :param obs: initial feedback from each game
        :param infos: additional infos of each game
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
        self._positive_scores = 0
        self._negative_scores = 0
        if self.hp.append_objective_to_tj:
            objective = infos[INFO_KEY.objective][0]
            if (self.game_id in self._loaded_objectives
                    and self._loaded_objectives[self.game_id][0] == objective):
                self._objective = self._loaded_objectives[self.game_id][1]
                self.info(
                    "substitute objective from ({}) to ({})".format(
                        objective, self._objective))
            else:
                self._objective = objective
            self._objective_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(self._objective))
        else:  # make sure no objective available
            self._objective = ""
            self._objective_ids = []

        if self.hp.walkthrough_guided_exploration:
            self._walkthrough = infos[INFO_KEY.walkthrough][0]
            self._continue_walkthrough = True

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
        self.info(report_status(
            [("training", self.is_training),
             ("#steps", self.in_game_t),
             ("score", scores[0]),
             ("positive scores", self._positive_scores),
             ("negative scores", self._negative_scores),
             ("won", infos[INFO_KEY.won][0]),
             ("lost", infos[INFO_KEY.lost][0]),
             ("policy_to_action", self.hp.policy_to_action),
             ("eps", self.eps),
             ("sampling_temp", self.hp.policy_sampling_temp)]))
        self._episode_has_started = False
        self._last_action_mask = None
        self._last_sys_action_mask = None
        self.game_id = None
        self._last_action = None
        self._cumulative_penalty = 0
        self._prev_last_action = None
        self._prev_master = None
        self._objective = ""
        self._objective_ids = []
        self._walkthrough = []
        self._continue_walkthrough = False

    def _delete_stale_context_objs(self) -> None:
        valid_tags = self._get_compatible_snapshot_tag()
        if len(valid_tags) > self.hp.max_snapshot_to_keep:
            self._stale_tags = list(reversed(sorted(
                valid_tags)))[self.hp.max_snapshot_to_keep:]
            self.info("tags to be deleted: {}".format(self._stale_tags))
            for tag in self._stale_tags:
                remove(self._get_context_obj_path_w_tag(self.memo_prefix, tag))
                remove(self._get_context_obj_path_w_tag(self.tjs_prefix, tag))
                remove(
                    self._get_context_obj_path_w_tag(self.action_prefix, tag))
                remove(self._get_context_obj_path_w_tag(self.fp_prefix, tag))
                remove(self._get_context_obj_path_w_tag(self.stc_prefix, tag))

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

    def _get_compatible_snapshot_tag(self) -> List[int]:
        action_tags = get_path_tags(self.model_dir, self.action_prefix)
        memo_tags = get_path_tags(self.model_dir, self.memo_prefix)
        tjs_tags = get_path_tags(self.model_dir, self.tjs_prefix)
        fp_tags = get_path_tags(self.model_dir, self.fp_prefix)
        stc_tags = get_path_tags(self.model_dir, self.stc_prefix)

        valid_tags = set(action_tags)
        valid_tags.intersection_update(memo_tags)
        valid_tags.intersection_update(tjs_tags)
        valid_tags.intersection_update(fp_tags)
        valid_tags.intersection_update(stc_tags)

        return list(valid_tags)

    def _is_time_to_save(self) -> bool:
        trained_steps = self.total_t - self.hp.observation_t + 1
        return (trained_steps % self.hp.save_gap_t == 0) and (trained_steps > 0)

    def _go_with_floor_plan(self, actions: List[str]) -> List[str]:
        """
        Update go-cardinal actions into go-room actions, if floor plan exists
        :param actions:
        :return:
        """
        local_map = self.floor_plan.get_map(self._curr_place)
        return (
            ["{} to {}".format(a, local_map.get(a))
             if a in local_map else a for a in actions])

    def _random_walk_for_collecting_fp(
            self, actions: List[str]) -> Optional[ActionDesc]:
        cardinal_go = list(filter(
            lambda a: a.startswith("go") and len(a.split()) == 2, actions))

        if (self.hp.collect_floor_plan and self.in_game_t < 50
                and len(cardinal_go) != 0):
            # collecting floor plan by choosing random action
            # if there are still go actions without room name
            # Notice that only choosing "go" actions cannot finish
            # collecting floor plan because there is the need to open doors
            # Notice also that only allow random walk in the first 50 steps
            open_actions = list(
                filter(lambda a: a.startswith("open"), actions))
            admissible_actions = cardinal_go + open_actions
            action = np.random.choice(admissible_actions)
            action_idx = self.actor.action2idx.get(action)
            action_desc = ActionDesc(
                action_type=ACT_TYPE.rnd_walk, action_idx=action_idx,
                token_idx=self.actor.action_matrix[action_idx],
                action_len=self.actor.action_len[action_idx],
                action=action, q_actions=None)
            return action_desc
        else:
            return None

    def _get_policy_action(self, action_mask: np.ndarray) -> ActionDesc:
        trajectory = self.tjs.fetch_last_state()
        state = self.stc.fetch_last_state()[-1]

        q_actions = self.core.policy(
            trajectory, state,
            self.actor.action_matrix, self.actor.action_len, action_mask)

        self.debug("q_actions: {}".format(list(q_actions)))
        self.debug("exp of q_actions: {}".format(list(np.exp(q_actions))))
        self.debug("ent: {:.5f}".format(
            entropy(pk=np.exp(q_actions), qk=np.ones_like(q_actions))))

        if self.hp.policy_to_action.lower() == "Sampling".lower():
            masked_action_idx = categorical_without_replacement(
                logits=q_actions / self.hp.policy_sampling_temp, k=1)
        elif self.hp.policy_to_action.lower() == "LinUCB".lower():
            cnt_action_array = []
            for mid in action_mask:
                cnt_action_array.append(self._cnt_action.get(mid, 0.))
            masked_action_idx, _ = get_best_1d_q(q_actions - cnt_action_array)
        elif self.hp.policy_to_action.lower() == "EPS".lower():
            masked_action_idx, _ = get_best_1d_q(q_actions)
        else:
            raise ValueError("Unknown policy utilization method: {}".format(
                self.hp.policy_to_action))

        action_idx = action_mask[masked_action_idx]
        action_desc = ActionDesc(
            action_type=ACT_TYPE.policy_drrn,
            action_idx=action_idx,
            token_idx=self.actor.action_matrix[action_idx],
            action_len=self.actor.action_len[action_idx],
            action=self.actor.actions[action_idx],
            q_actions=q_actions)
        return action_desc

    def _rule_based_policy(
            self, actions: List[str], instant_reward: float
    ) -> Optional[ActionDesc]:
        return None

    def _walkthrough_policy(self, actions: List[str]) -> Optional[ActionDesc]:
        if (not self.is_training
                or not self._continue_walkthrough
                or not self.hp.walkthrough_guided_exploration
                or len(self._walkthrough) == 0
                or not self.in_game_t < len(self._walkthrough)):
            return None

        # allow 1 complete walkthrough per 100 episodes
        if random.random() > self._walkthrough_prob_per_step(
                n_steps=len(self._walkthrough), prob_complete_play=0.01):
            self._continue_walkthrough = False
            self.info(
                "disallow walkthrough after {}/{} steps".format(
                    self.in_game_t, len(self._walkthrough)))

        gold_action = self._walkthrough[self.in_game_t]
        gold_action = self._go_with_floor_plan([gold_action])[0]
        assert gold_action in actions, "gold action is not in available actions"
        gold_action_idx = self.actor.action2idx.get(gold_action)
        action_desc = ActionDesc(
            action_type=ACT_TYPE.walkthrough,
            action_idx=gold_action_idx,
            action_len=self.actor.action_len[gold_action_idx],
            token_idx=self.actor.action_matrix[gold_action_idx],
            action=gold_action,
            q_actions=None)
        return action_desc

    def _choose_action(
            self,
            actions: List[str],
            action_mask: np.ndarray,
            instant_reward: float) -> ActionDesc:

        """
        Choose action, w.r.t.
         1) walkthrough guided action (training only)
         2) rule-based policy
         3) random walk floor map collection
         4) eps-greedy

        Args:
            actions: effective actions
            action_mask: mask ids of the effective actions
            instant_reward: the latest instant reward

        Returns:
            Action description
        """

        action_desc = self._walkthrough_policy(actions)
        if not action_desc:
            action_desc = self._rule_based_policy(actions, instant_reward)
            if not action_desc:
                action_desc = self._random_walk_for_collecting_fp(actions)
                if not action_desc:
                    action_desc = (
                        self._get_a_random_action(action_mask)
                        if random.random() < self.eps else
                        self._get_policy_action(action_mask))

        if self.hp.always_compute_policy and action_desc.q_actions is None:
            action_desc.q_actions = (
                self._get_policy_action(action_mask).q_actions)

        return action_desc

    def _get_raw_instant_reward(self, score: float) -> float:
        """raw instant reward between two consecutive scores"""
        instant_reward = score - self._cumulative_score
        self._cumulative_score = score
        return instant_reward

    def _get_repetition_penalty(
            self, master: str, raw_instant_reward: float) -> float:
        """
        add a penalty of self._cumulative_penalty if the current Action-Master
        repeats the failure of last Action-Master.
        """
        if (master == self._prev_master and self._last_action is not None
                and self._last_action.action == self._prev_last_action and
                raw_instant_reward <= 0):
            self._cumulative_penalty -= 0.1
        else:
            self._prev_last_action = (
                self._last_action.action
                if self._last_action is not None else None)
            self._prev_master = master
            self._cumulative_penalty = 0.
        return self._cumulative_penalty

    def _get_instant_reward(
            self, score: float, master: str, is_terminal: bool,
            won: bool, lost: bool) -> float:
        # there are three scenarios of game termination
        # 1. you won      --> encourage this action
        # 2. you lost     --> discourage this action
        # 3. out of step  --> do nothing
        raw_instant_reward = self._get_raw_instant_reward(score)
        if raw_instant_reward >= 0:
            self._positive_scores += raw_instant_reward
        else:
            self._negative_scores += raw_instant_reward
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
            curr_cumulative_penalty = self._get_repetition_penalty(
                master, raw_instant_reward)
            instant_reward += (curr_cumulative_penalty + (-0.1))
        instant_reward = self._clip_reward(instant_reward)
        return instant_reward

    @property
    def positive_scores(self):
        """
        Total positive scores earned
        """
        return self._positive_scores

    @property
    def negative_scores(self):
        """
        Total negative scores
        """
        return self._negative_scores

    def _collect_floor_plan(self, master: str, prev_place: str) -> str:
        """
        collect floor plan with latest master.
        if the current place doesn't match the previous place, and a go action
        is used to get the master, then we need to update the floor plan.

        :param master:
        :param prev_place: the name of previous place
        :return: the name of current place
        """
        room_name = self._get_room_name(master)
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

    def _train_one_batch(self) -> None:
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

        if self._is_time_to_save():
            self.save_snapshot()
            self.core.save_model()
            self.core.create_or_reload_target_model()

    def _clean_stale_context(self, tids: List[int]) -> None:
        self.tjs.request_delete_keys(tids)
        self.stc.request_delete_keys(tids)

    def _update_status(
            self, obs: List[str], scores: List[float], dones: List[bool],
            infos: Dict[str, List[Any]]) -> Tuple[str, float]:
        desc = infos[INFO_KEY.desc][0]
        master = (remove_zork_version_info(desc)
                  if self.in_game_t == 0 and desc else obs[0])
        instant_reward = self._get_instant_reward(
            scores[0], obs[0], dones[0],
            infos[INFO_KEY.won][0], infos[INFO_KEY.lost][0])

        if self._last_action:
            status = report_status([
                ("t", self.total_t),
                ("in_game_t", self.in_game_t),
                ("action", colored(
                    self._last_action.action,
                    "yellow"
                    if self._last_action.action_type == ACT_TYPE.policy_drrn
                    else "red", attrs=["underline"])),
                ("master", colored(
                    master.replace("\n", " ").strip(),
                    "cyan", attrs=["underline"])),
                ("reward", colored(
                    "{:.2f}".format(instant_reward),
                    "green" if instant_reward > 0 else "red")),
                ("is_terminal", dones[0])])
            if instant_reward > 0 or dones[0]:
                self.info(status)
            else:
                self.debug(status)
        else:
            self.info(report_status([
                ("master", colored(
                    master.replace("\n", " "), "cyan", attrs=["underline"])),
                ("max_score", infos[INFO_KEY.max_score][0])
            ]))

        if self.hp.collect_floor_plan:
            self._prev_place = self._curr_place
            self._curr_place = self._collect_floor_plan(
                master, self._prev_place)

        return master, instant_reward

    def _prepare_actions(self, admissible_actions: List[str]) -> List[str]:
        if self.hp.collect_floor_plan:
            effective_actions = self._go_with_floor_plan(admissible_actions)
        else:
            effective_actions = admissible_actions
        return effective_actions

    def _collect_new_sample(
            self, master: str, instant_reward: float, dones: List[bool],
            infos: Dict[str, List[Any]]) -> Tuple[
            List[str], np.ndarray, np.ndarray, float]:

        master_tokens = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(master))
        if self._last_action is not None:
            if self.hp.action_padding_in_tj:
                action_tokens = list(self._last_action.token_idx)
            else:  # trim action ids to its actual length
                action_tokens = list(
                    self._last_action.token_idx[:self._last_action.action_len])
        else:
            action_tokens = []

        self.tjs.append(ActionMaster(
            action_ids=action_tokens,
            master_ids=master_tokens,
            objective_ids=self._objective_ids,
            action=self._last_action.action if self._last_action else "",
            master=master))

        state = ObsInventory(
            obs=infos[INFO_KEY.desc][0],
            inventory=infos[INFO_KEY.inventory][0],
            sid=self.tjs.get_last_sid(),
            hs=get_hash_state(
                infos[INFO_KEY.desc][0], infos[INFO_KEY.inventory][0]))
        self.stc.append(state)

        admissible_actions = self._get_admissible_actions(infos)
        sys_action_mask = self.actor.extend(admissible_actions)
        effective_actions = self._prepare_actions(admissible_actions)
        action_mask = self.actor.extend(effective_actions)

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
            original_data = self.memo.append(memo_let)
            if isinstance(original_data, Memolet):
                if original_data.is_terminal:
                    self._stale_tids.append(original_data.tid)

        return effective_actions, action_mask, sys_action_mask, instant_reward

    def _next_step_action(
            self,
            actions: List[str],
            instant_reward: float,
            action_mask: np.ndarray,
            sys_action_mask: np.ndarray) -> str:

        if self.is_training:
            self.eps = self.eps_getter.eps(self.total_t - self.hp.observation_t)
        else:
            pass
        self._last_action = self._choose_action(
            actions, action_mask, instant_reward)
        action = self._last_action.action
        action_idx = self._last_action.action_idx

        if self._last_action.action_type == ACT_TYPE.policy_drrn:
            if action_idx not in self._cnt_action:
                self._cnt_action[action_idx] = 0.
            self._cnt_action[action_idx] += 0.1

        self._last_action_mask = action_mask
        self._last_sys_action_mask = sys_action_mask
        # revert back go actions for the game playing
        # TODO: better collecting floor plan schedule?
        # TODO: better condition for reverting go-actions?
        if (self.hp.collect_floor_plan and action.startswith("go")
                and "to" in action):
            action = " ".join(action.split()[:2])
        return action

    def act(self, obs: List[str], scores: List[int], dones: List[bool],
            infos: Dict[str, List[Any]]) -> Optional[List[str]]:
        """
        Acts upon the current list of observations.
        One text command must be returned for each observation.

        Args:
            obs: observed texts for each game
            scores: score obtained so far for each game
            dones: whether a game is finished
            infos: extra information requested from TextWorld

        Returns:
            if all dones, return None, else return actions

        Notes:
            Commands returned for games marked as `done` have no effect.
            The states for finished games are simply copy over until all
            games are done.
        """
        if not self._episode_has_started:
            self._start_episode(obs, infos)

        assert len(obs) == 1, "cannot handle batch game training"
        master, instant_reward = self._update_status(obs, scores, dones, infos)
        (actions, action_mask, sys_action_mask, instant_reward
         ) = self._collect_new_sample(master, instant_reward, dones, infos)
        # notice the position of all(dones)
        # make sure add the last action-master pair into memory
        if all(dones):
            self._end_episode(obs, scores, infos)
            return None

        player_t = self._next_step_action(
            actions, instant_reward, action_mask, sys_action_mask)
        if self.is_training and self.total_t >= self.hp.observation_t:
            self._train_one_batch()
        self.total_t += 1
        self.in_game_t += 1
        return [player_t] * len(obs)
