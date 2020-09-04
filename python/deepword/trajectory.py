import time
import numpy as np
from typing import List, Optional, Dict
from typing import TypeVar, Generic

from deepword.utils import eprint


T = TypeVar('T')


class Trajectory(Generic[T]):
    """
    BaseTrajectory only takes care of interacting with Agent on collecting
    game scripts.
    Fetching data from Trajectory feeding into Encoder should be implemented
    in extended classes.
    """
    def __init__(self, num_turns: int, size_per_turn: int = 1) -> None:
        """
        Take the ActionMaster (AM) as an example,
        Trajectory(AM1, AM2, AM3, AM4, AM5), and last_sid points to AM5;
        num_turns = 1 means we choose [AM4, AM5];

        size_per_turn only controls the way we separate pre- and post-trajectory
        default with size_per_turn = 1, AM4 is the pre-trajectory of AM5.

        Sometimes we need to change it, e.g. with legacy data where we store
        trajectory as Trajectory(M1, A1, M2, A2, M3, A3, M4), and the last
        sid points to M4, then the pre-trajectory of [A3, M4] is [A2, M3],
        that's why the size_per_turn should set to 2.

        :param num_turns: how many turns to choose other than current turn
        :param size_per_turn: how many cells count as one turn
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
        :return:
        """
        return len(self.curr_tj) - 1

    def get_current_tid(self) -> int:
        return self.curr_tid

    def request_delete_keys(self, ks: List[int]) -> None:
        if not ks:
            return
        for k in sorted(self.trajectories.keys()):
            if k > max(ks):
                break
            else:
                self.trajectories.pop(k, None)
                eprint('trajectory {} (time<=) {} deleted'.format(k, max(ks)))

    @classmethod
    def _ctime(cls) -> int:
        return int(round(time.time() * 1000))

    def add_new_tj(self, tid: Optional[int] = None) -> int:
        if tid is None or tid in self.trajectories:
            tid = self._ctime()
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

        :param content:
        :return:
        """
        self.curr_tj.append(content)

    def save_tjs(self, path: str) -> None:
        tids = list(self.trajectories.keys())
        vals = list(self.trajectories.values())
        np.savez(
            path,
            tids=tids,
            vals=vals + [[None]],
            curr_tid=[self.curr_tid],
            curr_tj=[self.curr_tj] + [[None]])

    def load_tjs(self, path: str) -> None:
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
        if tid == self.curr_tid:
            tj = self.curr_tj
        elif tid in self.trajectories:
            tj = self.trajectories[tid]
        else:
            return []
        state = tj[max(0, sid - self.num_turns):sid + 1]
        return state

    def fetch_last_state(self) -> List[T]:
        return self.fetch_state_by_idx(self.curr_tid, self.get_last_sid())

    def fetch_batch_states(
            self, b_tid: List[int], b_sid: List[int]) -> List[List[T]]:
        batch_states = []
        for tid, sid in zip(b_tid, b_sid):
            batch_states.append(self.fetch_state_by_idx(tid, sid))
        return batch_states

    def fetch_batch_pre_states(
            self, b_tid: List[int], b_sid: List[int]) -> List[List[T]]:
        batch_states = []
        for tid, sid in zip(b_tid, b_sid):
            batch_states.append(
                self.fetch_state_by_idx(tid, sid - self.size_per_turn))
        return batch_states
