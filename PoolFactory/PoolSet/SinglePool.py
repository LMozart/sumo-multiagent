import numpy as np
from PoolFactory.Pool import ExperiencePool


class SinglePool(ExperiencePool):
    def __init__(self, numb_agent, batch_size, max_size):
        ExperiencePool.__init__(self, numb_agent, batch_size, max_size)
        self.max_reward = 10
        self.min_reward = -10
        self.ExperiencePool = []

    def store(self, experiences):
        if self.experience_size == 0:
            self.ExperiencePool = experiences[0]
            self.experience_size += 1
        else:
            for i in range(len(experiences)):
                self.ExperiencePool['s'] = np.insert(self.ExperiencePool['s'], 0,
                                                     values=experiences[i]['s'], axis=0)
                self.ExperiencePool['r'] = np.insert(self.ExperiencePool['r'], 0,
                                                     values=experiences[i]['r'], axis=0)
                self.ExperiencePool['a'] = np.insert(self.ExperiencePool['a'], 0,
                                                     values=experiences[i]['a'], axis=0)
                self.ExperiencePool['next_s'] = np.insert(self.ExperiencePool['next_s'], 0,
                                                          values=experiences[i]['next_s'], axis=0)
                self.ExperiencePool['terminal'] = np.insert(self.ExperiencePool['terminal'], 0,
                                                            values=experiences[i]['terminal'], axis=0)
                self.experience_size += 1
        for i in range(len(experiences)):
            if experiences[i]['r'] > self.max_reward:
                self.max_reward = experiences[i]['r']
            if experiences[i]['r'] < self.min_reward:
                self.min_reward = experiences[i]['r']

    def sample_index(self):
        index = np.random.randint(0, self.experience_size, size=self.batch_size)
        return index

    def fetch_sample_experience(self, index):
        o_states = np.array(self.ExperiencePool['s'])[index]
        o_actions = np.array(self.ExperiencePool['a'])[index]
        o_next_states = np.array(self.ExperiencePool['next_s'])[index]
        o_rewards = np.array(self.ExperiencePool['r'])[index] / np.abs(self.min_reward)
        # if self.max_reward > np.abs(self.min_reward):
        #     factor = self.max_reward
        # else:
        #     factor = np.abs(self.min_reward)
        # o_rewards = np.array(self.ExperiencePool['r'])[index] / factor
        o_done = np.array(self.ExperiencePool['terminal'])[index]
        if self.experience_size > self.max_experience_size:
            self._clip()
        return o_states, o_next_states, o_actions, o_rewards, o_done
