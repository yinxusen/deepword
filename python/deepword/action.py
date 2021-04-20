from typing import List, Dict, Optional, Tuple

import numpy as np

from deepword.log import Logging
from deepword.tokenizers import Tokenizer


class ActionCollector(Logging):
    """
    Collect actions for different games
    """
    def __init__(
            self, tokenizer: Tokenizer, n_tokens: int, unk_val_id: int,
            padding_val_id: int) -> None:
        """
        Args:
            tokenizer: see :py:mod:`deepword.tokenizers`
            n_tokens: max allowed number of tokens for all actions
            unk_val_id: ID of the unknown token
            padding_val_id: ID of the padding token
        """
        super(ActionCollector, self).__init__()
        # collections of all actions and its indexed vectors
        self._actions_base: Dict[str, List[str]] = dict()
        self._action_matrix_base: Dict[str, List[np.ndarray]] = dict()
        self._action_len_base: Dict[str, List[int]] = dict()

        # metadata of the action collector
        self._n_tokens: int = n_tokens
        self._unk_val_id: int = unk_val_id
        self._padding_val_id: int = padding_val_id
        self._tokenizer: Tokenizer = tokenizer

        # current episode actions
        self._action2idx = None
        self._actions = None
        self._curr_aid = 0
        self._curr_gid = None
        self._action_matrix = None
        self._action_len = None

    def _reset_episode_vars(self) -> None:
        self._action2idx: Dict[str, int] = {}
        self._actions: List[str] = []
        # aid always points to the next future action
        self._curr_aid: int = 0
        self._curr_gid: Optional[str] = None
        self._action_matrix: List[np.ndarray] = []
        self._action_len: List[int] = []

    def add_new_episode(self, gid: str) -> None:
        """
        Add a new episode with game ID.

        Args:
            gid: game ID, a string that separate different games.
        """
        if gid == self._curr_gid:
            return

        if self._size != 0 and self._curr_gid is not None:
            self._actions_base[self._curr_gid] = self._actions
            self._action_matrix_base[self._curr_gid] = self._action_matrix
            self._action_len_base[self._curr_gid] = self._action_len

        self._reset_episode_vars()
        self._curr_gid = gid
        if self._curr_gid in self._actions_base:
            self.info("found existing episode: {}".format(self._curr_gid))
            self._curr_aid = len(self._actions_base[self._curr_gid])
            self._actions = self._actions_base[self._curr_gid]
            self._action_matrix = self._action_matrix_base[self._curr_gid]
            self._action_len = self._action_len_base[self._curr_gid]
            self._action2idx = dict(
                [(a, i) for (i, a) in enumerate(self._actions)])
            self.info("{} actions loaded".format(self._size))

    def _convert_action_to_ids(self, action: str) -> Tuple[np.ndarray, int]:
        token_ids = self._tokenizer.convert_tokens_to_ids(
            self._tokenizer.tokenize(action))
        action_len = min(self._n_tokens, len(token_ids))
        action_idx = np.zeros(self._n_tokens, dtype=np.int32)
        action_idx[:action_len] = token_ids[:action_len]
        return action_idx, action_len

    def extend(self, actions: List[str]) -> np.ndarray:
        """
        Extend actions into ActionCollector.

        Args:
            actions: a list of actions for current episode of game-playing.

        Returns:
            sorted mask ids
        """
        mask_idx = []
        for a in actions:
            if a not in self._action2idx:
                self._action2idx[a] = self._curr_aid
                action_idx, action_len = self._convert_action_to_ids(a)
                self._action_len.append(action_len)
                self._action_matrix.append(action_idx)
                self._actions.append(a)
                self._curr_aid += 1
            mask_idx.append(self._action2idx[a])
        return np.asarray(sorted(mask_idx))

    def get_action_matrix(self, gid: Optional[str] = None) -> np.ndarray:
        """
        Get action matrix for a game.

        Args:
            gid: the game ID. `None` falls back to current activated game.

        Returns:
            an array of actions, each action is a vector of its IDs to tokens
             that are filled with padding in the end to reach the same token
             size.
        """
        if gid is None or gid == self._curr_gid:
            return self.action_matrix
        else:
            return np.asarray(self._action_matrix_base[gid])

    @property
    def action_matrix(self) -> np.ndarray:
        """
        Current action matrix
        """
        return np.asarray(self._action_matrix)

    @property
    def action2idx(self) -> Dict[str, int]:
        """
        Current action to token IDs
        """
        return self._action2idx

    def get_action_len(self, gid: Optional[str] = None) -> np.ndarray:
        """
        Get action lengths for a game

        Args:
            gid: game ID, `None` falls back to current episode of game.

        Returns:
            an array of int, each element is a length for that action
        """
        if gid is None or gid == self._curr_gid:
            return self.action_len
        else:
            return np.asarray(self._action_len_base[gid])

    @property
    def action_len(self) -> np.ndarray:
        """
        Current action lengths
        """
        return np.asarray(self._action_len)

    def get_actions(self, gid: Optional[str] = None) -> List[str]:
        """
        Get all actions for a game.

        Args:
            gid: game ID, `None` falls back to current episode of game.

        Returns:
            a list of actions in string
        """
        if gid is None or gid == self._curr_gid:
            return self._actions
        else:
            return self._actions_base[gid]

    def get_game_ids(self) -> List[str]:
        """
        Get all game IDs in this ActionCollector.
        """
        return list(self._actions_base.keys())

    @property
    def actions(self) -> List[str]:
        """
        Current actions in string.
        """
        return self._actions

    @property
    def _size(self) -> int:
        return self._curr_aid

    def save_actions(self, path: str) -> None:
        """
        Save all actions to a path as a npz file.

        Args:
            path: a npz path to save
        """
        if self._size != 0 and self._curr_gid is not None:
            self._actions_base[self._curr_gid] = self._actions
        actions_base_keys = list(self._actions_base.keys())
        actions_base_vals = list(self._actions_base.values())
        np.savez(
            path,
            actions_base_keys=actions_base_keys,
            actions_base_vals=actions_base_vals,
            action_matrix=[self._action_matrix_base])

    def load_actions(self, path: str) -> None:
        """
        Load all actions in this ActionCollector

        Args:
            path: a path to a npz file.
        """
        saved = np.load(path, allow_pickle=True)
        actions_base = dict(
            zip(list(saved["actions_base_keys"]),
                list(saved["actions_base_vals"])))
        for gid in actions_base:
            self.add_new_episode(gid)
            self.extend(actions_base[gid])
