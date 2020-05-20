import numpy as np
import random
from PoolFactory.Pool import ExperiencePool


class PERSinglePool(ExperiencePool):
    def __init__(self, numb_agent, batch_size, max_size):
        ExperiencePool.__init__(self, numb_agent, batch_size, max_size)
        self.max_reward = 10
        self.min_reward = -10
        self.ExperiencePool = SumTree(max_size)
        self.alpha = 0.5
        self.epsilon = 0.01

    def store(self, experiences):
        for i in range(len(experiences)):
            s = experiences[i]['s']
            r = experiences[i]['r']
            a = experiences[i]['a']
            d = experiences[i]['terminal']
            n_s = experiences[i]['next_s']
            error = experiences[i]['td']
            experience = (s, a, r, d, n_s)
            priority = self.priority(error[0])
            self.ExperiencePool.add(priority, experience)
            self.experience_size += 1
        for i in range(len(experiences)):
            if experiences[i]['r'] > self.max_reward:
                self.max_reward = experiences[i]['r']
            if experiences[i]['r'] < self.min_reward:
                self.min_reward = experiences[i]['r']

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def sample_index(self):
        index = np.random.randint(0, self.experience_size, size=self.batch_size)
        return index

    def fetch_sample_experience(self, index):
        batch = []
        T = self.ExperiencePool.total() // self.batch_size
        for i in range(self.batch_size):
            a, b = T * i, T * (i + 1)
            s = random.uniform(a, b)
            idx, error, data = self.ExperiencePool.get(s)
            batch.append((data, idx))
        o_index = np.squeeze(np.array([i[1] for i in batch]))
        o_states = np.squeeze(np.array([i[0][0] for i in batch]))
        o_actions = np.squeeze(np.array([i[0][1] for i in batch]))
        o_rewards = np.squeeze(np.array([i[0][2] / np.abs(self.min_reward) for i in batch]))
        o_done = np.squeeze(np.array([i[0][3] for i in batch]))
        o_next_states = np.squeeze(np.array([i[0][4] for i in batch]))
        return o_states, o_next_states, o_actions, o_rewards, o_done, o_index

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.ExperiencePool.update(idx, self.priority(new_error))


""" Original Code by @jaara: https://github.com/jaara/AI-blog/blob/master/SumTree.py
"""


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]
