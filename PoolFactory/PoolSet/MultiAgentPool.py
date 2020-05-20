import numpy as np
from PoolFactory.Pool import ExperiencePool


class MultiAgentPool(ExperiencePool):
    def __init__(self, numb_agent, batch_size, max_size):
        ExperiencePool.__init__(self, numb_agent, batch_size, max_size)
        self.max_reward = 0
        self.min_reward = -1

    def store(self, experiences):
        if self.experience_size == 0:
            for i in range(len(experiences)):
                self.ExperiencePool[i] = experiences[i]

        else:
            for i in range(len(experiences)):
                self.ExperiencePool[i]['s'] = np.insert(self.ExperiencePool[i]['s'], 0,
                                                        values=experiences[i]['s'], axis=0)
                self.ExperiencePool[i]['r'] = np.insert(self.ExperiencePool[i]['r'], 0,
                                                        values=experiences[i]['r'], axis=0)
                self.ExperiencePool[i]['a'] = np.insert(self.ExperiencePool[i]['a'], 0,
                                                        values=experiences[i]['a'], axis=0)
                self.ExperiencePool[i]['next_s'] = np.insert(self.ExperiencePool[i]['next_s'], 0,
                                                             values=experiences[i]['next_s'], axis=0)
                self.ExperiencePool[i]['terminal'] = np.insert(self.ExperiencePool[i]['terminal'], 0,
                                                               values=experiences[i]['terminal'], axis=0)
        for i in range(len(experiences)):
            if experiences[i]['r'] > self.max_reward:
                self.max_reward = experiences[i]['r']
            if experiences[i]['r'] < self.min_reward:
                self.min_reward = experiences[i]['r']

        self.experience_size += 1

    def sample_index(self):
        index = np.random.randint(0, self.experience_size, size=self.batch_size)
        return index

    def fetch_sample_experience(self, index):
        o_states = []
        o_actions = []
        o_rewards = []
        o_next_states = []
        o_done = []
        first = True
        for x in range(len(self.ExperiencePool)):
            o_states.append(np.array(self.ExperiencePool[x]['s'])[index])
            o_actions.append(np.array(self.ExperiencePool[x]['a'])[index])
            o_next_states.append(np.array(self.ExperiencePool[x]['next_s'])[index])
            if first:
                # np.array(self.ExperiencePool[x]['r'])[index].shape) == (64, 1)
                o_rewards = -1 * np.array(self.ExperiencePool[x]['r'])[index] / self.min_reward
                o_done = np.array(self.ExperiencePool[x]['terminal'])[index]
                first = False
            else:
                # np.array(self.ExperiencePool[x]['r'])[index].T.shape == (1, 64)
                o_rewards = np.insert(o_rewards, o_rewards.shape[1],
                                      values=(-1 * np.array(self.ExperiencePool[x]['r'])[index].T / self.min_reward),
                                      axis=1)
        if self.experience_size > self.max_experience_size:
            self._clip()
        return o_states, o_next_states, o_actions, o_rewards, o_done
