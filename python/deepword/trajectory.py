from typing import List, Optional, Dict
from typing import TypeVar, Generic

import numpy as np

from deepword.log import Logging
from deepword.utils import ctime, report_status

T = TypeVar('T')


class Trajectory(Generic[T], Logging):
    """
    BaseTrajectory only takes care of interacting with Agent on collecting
    game scripts.
    Fetching data from Trajectory feeding into Encoder should be implemented
    in extended classes.
    """

    def __init__(self, num_turns: int, size_per_turn: int = 1):
        """
        Take the ActionMaster (AM) as an example,
        Trajectory(AM1, AM2, AM3, AM4, AM5), and last_sid points to AM5;
        num_turns = 1 means we choose [AM5];

        size_per_turn only controls the way we separate pre- and post-trajectory
        default with size_per_turn = 1, AM4 is the pre-trajectory of AM5.

        Sometimes we need to change it, e.g. with legacy data where we store
        trajectory as Trajectory(M1, A1, M2, A2, M3, A3, M4), and the last
        sid points to M4, then the pre-trajectory of [A3, M4] is [A2, M3],
        that's why the size_per_turn should set to 2.

        Args:
            num_turns: how many turns to choose other than current turn
            size_per_turn: how many cells count as one turn
        """

        super(Trajectory, self).__init__()
        self.trajectories: Dict[int, List[T]] = dict()
        self.curr_tj: Optional[List[T]] = None
        self.curr_tid: Optional[int] = None
        self.num_turns: int = num_turns
        self.size_per_turn: int = size_per_turn

    def get_last_sid(self) -> int:
        """
        A state is defined as a series of interactions between a game and an
        agent ended with the game's last response.
        e.g. "G0, A1, G2, A3, G4" is a state ended with the game's last response
        named "G4".
        """

        return len(self.curr_tj) - 1

    def get_current_tid(self) -> int:
        """
        Get current trajectory id
        """

        return self.curr_tid

    def request_delete_keys(self, ks: List[int]) -> None:
        """
        Request to delete all trajectories with keys in `ks`.

        Args:
            ks: a list of keys of trajectories to be deleted
        """

        if not ks:
            return
        for k in sorted(self.trajectories.keys()):
            if k > max(ks):
                break
            else:
                self.trajectories.pop(k, None)
                self.debug(
                    'trajectory {} (time<=) {} deleted'.format(k, max(ks)))

    def add_new_tj(self, tid: Optional[int] = None) -> int:
        """
        Add a new trajectory

        Args:
            tid: trajectory id, `None` falls back to auto-generated id.

        Returns:
            a tid
        """

        if tid is None or tid in self.trajectories:
            tid = ctime()
        if self.curr_tj is not None and len(self.curr_tj) != 0:
            self.trajectories[self.curr_tid] = self.curr_tj
        self.curr_tid = tid
        self.curr_tj = []
        return tid

    def append(self, content: T) -> None:
        """
        Use generic type for content
        the trajectory class doesn't care what has been stored
        and also doesn't process them

        Args:
            content: something to add in the current trajectory
        """

        self.curr_tj.append(content)

    def save_tjs(self, path: str) -> None:
        """
        Save all trajectories in a npz file

        All trajectory ids, all trajectories, current trajectory id, and
        current trajectory will be saved.
        """

        tids = list(self.trajectories.keys())
        vals = list(self.trajectories.values())
        np.savez(
            path,
            tids=tids,
            vals=vals + [[None]],
            curr_tid=[self.curr_tid],
            curr_tj=[self.curr_tj] + [[None]])

    def load_tjs(self, path: str) -> None:
        """
        Load trajectories from a npz file
        """

        tjs = np.load(path, allow_pickle=True)
        self.curr_tid = tjs["curr_tid"][0]
        self.curr_tj = list(tjs["curr_tj"][0])
        tids = tjs["tids"]
        vals = tjs["vals"]
        if len(tids) + 1 == len(vals):
            vals = vals[:-1]
        assert len(tids) == len(vals), "incompatible trajectory ids and values"
        for i in range(len(tids)):
            self.trajectories[tids[i]] = list(vals[i])

    def fetch_state_by_idx(self, tid: int, sid: int) -> List[T]:
        """
        fetch a state given trajectory id and state id

        Returns:
            a list of contents
        """

        if tid == self.curr_tid:
            tj = self.curr_tj
        elif tid in self.trajectories:
            tj = self.trajectories[tid]
        else:
            return []
        ss = max(0, sid - self.num_turns + 1)
        ee = sid + 1  # sid should be included
        state = tj[ss: ee]
        if not state:
            self.debug("empty trajectory:\n{}".format(
                report_status([
                    ("tid", tid), ("sid", sid), ("tj_len", len(tj)),
                    ("ss", ss), ("ee", ee)
                ])))
        return state

    def fetch_last_state(self) -> List[T]:
        """
        Fetch the last state from the current trajectory

        Returns:
            a list of contents
        """

        return self.fetch_state_by_idx(self.curr_tid, self.get_last_sid())

    def fetch_batch_states(
            self, b_tid: List[int], b_sid: List[int]) -> List[List[T]]:
        """
        Fetch a batch of states given trajectory ids and state ids.

        Args:
            b_tid: a batch of trajectory ids
            b_sid: a batch of state ids

        Returns:
            a list of lists of contents
        """

        batch_states = []
        for tid, sid in zip(b_tid, b_sid):
            batch_states.append(self.fetch_state_by_idx(tid, sid))
        return batch_states

    def fetch_batch_pre_states(
            self, b_tid: List[int], b_sid: List[int]) -> List[List[T]]:
        """
        Fetch a batch of pre-states given trajectory ids and state ids

        the position of pre-states is depend on `size_per_turn`.

        Args:
            b_tid: a batch of trajectory ids
            b_sid: a batch of state ids

        Returns:
            a list of lists of contents
        """
        batch_states = []
        for tid, sid in zip(b_tid, b_sid):
            batch_states.append(
                self.fetch_state_by_idx(tid, sid - self.size_per_turn))
        return batch_states
