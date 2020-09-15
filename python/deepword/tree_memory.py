"""
This SumTree code is modified version and the original code is from:
https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
"""

from typing import Tuple, List
from typing import TypeVar, Generic

import numpy as np

from deepword.log import Logging
from deepword.sum_tree import SumTree

E = TypeVar('E')  # type of the experience


class TreeMemory(Logging, Generic[E]):
    """
    TreeMemory to store and sample experiences to replay.
    """

    def __init__(self, capacity: int):
        super(TreeMemory, self).__init__()
        self.tree = SumTree(capacity)
        self.used_buffer_size = 0
        self.per_e = 0.01  # to avoid zero probability
        self.per_a = 0.6  # tradeoff between sampling ~ priority and random
        self.per_b = 0.4  # importance-sampling, increasing to 1
        self.per_b_inc_step = 0.001  # inc per_b per sampling
        self.abs_err_upper_bound = 1.  # clip the error

    def __len__(self) -> int:
        return self.used_buffer_size

    def append(self, experience: E) -> E:
        """
        New experiences have a score of max priority over all leaves to make
        sure to be sampled.

        Args:
            experience: a new experience

        Returns:
            previous experience in the same position
        """
        # find the max priority over leaves
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        max_priority = (
            self.abs_err_upper_bound if max_priority == 0 else max_priority)
        prev_data = self.tree.add(max_priority, experience)
        if self.used_buffer_size < self.tree.capacity:
            self.used_buffer_size += 1
        return prev_data

    def uniform_sample_batch(self, n: int) -> np.ndarray:
        """
        Randomly sample a batch of experiences

        Args:
            n: batch size

        Returns:
            a batch of experiences
        """
        return np.random.choice(
            self.tree.data[:-self.used_buffer_size], size=n)

    def sample_batch(self, n: int) -> Tuple[np.ndarray, List[E], np.ndarray]:
        """
        Sample a batch of experiences according to priority values

        - First, to sample a batch of n size, the range [0, priority_total] is
          divided into n equally ranges.
        - Then a value is uniformly sampled from each range
        - We search in the `SumTree`, the experience where priority score
          correspond to sample values are retrieved from.
        - Then, we calculate importance sampling (IS) weights
          for each element in the batch

        Args:
            n: batch size

        Returns:
            tree index, experiences, IS weights
        """
        memory_b = []
        b_idx = np.empty((n,), dtype=np.int32)
        b_is_weights = np.empty((n,), dtype=np.float32)

        # Calculate the priority segment
        priority_segment = self.tree.total_priority / n

        # Here we increasing the PER_b each time we sample a new batch
        self.per_b = min(1., self.per_b + self.per_b_inc_step)

        # Calculating the max_weight
        p_min = (np.min(
            self.tree.tree[-self.tree.capacity:][:self.used_buffer_size]) /
                 self.tree.total_priority)
        max_weight = np.power(p_min * self.used_buffer_size, -self.per_b)

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            # P(i)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_is_weights[i] = np.power(
                self.used_buffer_size * sampling_probabilities,
                -self.per_b) / max_weight
            b_idx[i] = index

            memory_b.append(data)

        return b_idx, memory_b, b_is_weights

    def batch_update(
            self, tree_idx: np.ndarray, abs_errors: np.ndarray) -> None:
        """
        Update the priorities on the tree

        Args:
            tree_idx: an array of index (int)
            abs_errors: an array of abs errors (float)
        """
        abs_errors += self.per_e  # avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper_bound)
        ps = np.power(clipped_errors, self.per_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def save_memo(self, path: str) -> None:
        """
        Save the memory to a npz file

        Args:
            path: path to a npz file
        """
        np.savez(
            path, tree=self.tree.tree, data=self.tree.data,
            data_pointer=self.tree.data_pointer,
            used_buffer_size=self.used_buffer_size)

    def load_memo(self, path: str) -> None:
        """
        load memo only apply to a new memo without any appending.
        load memo will change previous tree structure.

        Args:
            path: a npz file to load
        """
        memo = np.load(path, allow_pickle=True)
        self.tree.tree = memo["tree"]
        self.tree.data = memo["data"]
        self.tree.data_pointer = memo["data_pointer"]
        self.used_buffer_size = memo["used_buffer_size"]
