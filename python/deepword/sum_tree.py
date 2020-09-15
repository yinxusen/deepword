"""
This SumTree code is modified version of Morvan Zhou:
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
"""

from typing import Tuple
from typing import TypeVar, Generic

import numpy as np

E = TypeVar('E')  # type of the experience


class SumTree(Generic[E]):
    """
    The SumTree is a binary tree, with leaf nodes containing the real data.
    """
    def __init__(self, capacity: int):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        self.data_pointer = 0
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1)
        # Remember we are in a binary node (each node has max 2 children)
        # so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def __repr__(self) -> str:
        return str(self.tree[-self.capacity:])

    def add(self, priority: float, data: E):
        """
        Add an experience into the tree with a priority

        Args:
            priority: priority of sampling
            data: experience of the replay

        Returns:
            old data at the same position, 0 if unset
        """
        prev_data = self.data[self.data_pointer]
        self.data[self.data_pointer] = data

        # turn data_pointer to tree_index
        tree_index = self.data_pointer + self.capacity - 1
        # Update the leaf
        self.update(tree_index, priority)

        # data_pointer moves to the next position
        self.data_pointer += 1
        # If we're above the capacity, go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        return prev_data

    def update(self, tree_index: int, priority: float) -> None:
        """
        Update the leaf priority score and propagate the change through tree

        Args:
            tree_index: tree index of the current data_pointer
            priority: priority sampling value
        """
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree, until the root
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v: float) -> Tuple[int, float, E]:
        """
        Get a leaf_index w.r.t. a priority value
        the selected leaf_index must have the smallest priority among all leaves
        that have larger priority values than v.

        Args:
            v: a priority value

        Returns:
            leaf index, priority, and experience associated with the leaf index
        """
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self) -> float:
        """
        The total priority is the value on the root node.
        """
        return self.tree[0]
