import numpy as np

from deeptextworld.sum_tree import SumTree


class TreeMemory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)
        self.used_buffer_size = 0

    def __len__(self):
        return self.used_buffer_size

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def append(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        prev_data = self.tree.add(max_priority, experience)  # set the max p for new p
        if self.used_buffer_size < self.tree.capacity:
            self.used_buffer_size += 1
        return prev_data

    def uniform_sample_batch(self, n):
        return np.random.choice(self.tree.data[:-self.used_buffer_size], size=n)

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample_batch(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx = np.empty((n,), dtype=np.int32)
        b_ISWeights = np.empty((n,), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min(
            [1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = (np.min(
            self.tree.tree[-self.tree.capacity:][:self.used_buffer_size]) /
                 self.tree.total_priority)
        max_weight = np.power(p_min * self.used_buffer_size, -self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i] = np.power(
                self.used_buffer_size * sampling_probabilities,
                -self.PER_b) / max_weight
            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def save_memo(self, path):
        np.savez(
            path, tree=self.tree.tree, data=self.tree.data,
            data_pointer=self.tree.data_pointer,
            used_buffer_size=self.used_buffer_size)

    def load_memo(self, path):
        """
        load memo only apply to a new memo without any appending.
        load memo will change previous tree structure.
        :param path:
        :return:
        """
        memo = np.load(path)
        self.tree.tree = memo["tree"]
        self.tree.data = memo["data"]
        self.tree.data_pointer = memo["data_pointer"]
        self.used_buffer_size = memo["used_buffer_size"]
