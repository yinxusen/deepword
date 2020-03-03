import time
import numpy as np
from typing import List, Optional

from deeptextworld.log import Logging
from deeptextworld.utils import flatten
from deeptextworld.agents.utils import ActionMaster


class Trajectory(Logging):
    """
    BaseTrajectory only takes care of interacting with Agent on collecting
    game scripts.
    Fetching data from Trajectory feeding into Encoder should be implemented
    in extended classes.
    """
    def __init__(self, num_turns: int) -> None:
        super(Trajectory, self).__init__()
        self.trajectories = {}
        self.curr_tj = None
        self.curr_tid = None
        self.num_turns = num_turns

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
                self.debug(
                    'trajectory {} (time<=) {} deleted'.format(k, max(ks)))

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

    def append(self, action_master: ActionMaster) -> None:
        self.curr_tj.append(action_master)

    def save_tjs(self, path: str) -> None:
        tids = list(self.trajectories.keys())
        vals = list(self.trajectories.values())
        np.savez(
            path, tids=tids, vals=vals,
            curr_tid=[self.curr_tid], curr_tj=[self.curr_tj])

    def load_tjs(self, path: str) -> None:
        tjs = np.load(path, allow_pickle=True)
        self.curr_tid = tjs["curr_tid"][0]
        self.curr_tj = list(tjs["curr_tj"][0])
        tids = tjs["tids"]
        vals = tjs["vals"]
        assert len(tids) == len(vals), "incompatible trajectory ids and values"
        for i in range(len(tids)):
            self.trajectories[tids[i]] = list(vals[i])

    def fetch_state_by_idx(self, tid: int, sid: int) -> List[str]:
        if tid == self.curr_tid:
            tj = self.curr_tj
        elif tid in self.trajectories:
            tj = self.trajectories[tid]
        else:
            return []
        state = flatten(
            [list(am) for am in tj[max(0, sid - self.num_turns):sid + 1]])
        return state

    def fetch_last_state(self) -> List[str]:
        return self.fetch_state_by_idx(self.curr_tid, self.get_last_sid())

    def fetch_batch_states(self, b_tid, b_sid) -> List[List[str]]:
        batch_states = []
        for tid, sid in zip(b_tid, b_sid):
            batch_states.append(self.fetch_state_by_idx(tid, sid))
        return batch_states
