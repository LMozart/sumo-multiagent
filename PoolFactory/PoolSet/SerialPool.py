import numpy as np
from PoolFactory.Pool import ExperiencePool


class SerialPool(ExperiencePool):
    def __init__(self, numb_agent, batch_size, max_size):
        ExperiencePool.__init__(self, numb_agent, batch_size, max_size)
        self.max_reward = 0
        self.min_reward = -10
        self.ExperiencePool = []
        self.RewardBuffer = None

    def store_buffer(self, experiences):
        for i in range(len(experiences)):
                experiences[i]['r'] = self.RewardBuffer[i]
        self.store(experiences)

    def record_reward(self, ac_reward):
        self.RewardBuffer = ac_reward

    def store(self, experiences):
        # i is time step
        for i in range(len(experiences)):
            hist_s = []
            hist_n_s = []
            hist_a = []
            hist_r = float(experiences[i]['r'])
            hist_tmn = []
            for j in range(len(experiences[0])):
                hist_s.append(experiences[i][j]['s'])
                hist_n_s.append(experiences[i][j]['next_s'])
                hist_a.append(experiences[i][j]['a'])
                hist_tmn.append(experiences[i][j]['terminal'])

            hist_s = np.array(hist_s)[np.newaxis, :, :, :]
            hist_a = np.array(hist_a)[np.newaxis, :, :, :]
            hist_n_s = np.array(hist_n_s)[np.newaxis, :, :, :]
            hist_r = np.array(hist_r)[np.newaxis, :] # (5, )
            hist_tmn = np.array(hist_tmn)[np.newaxis, :, :, :] # (5,1,1)

            if self.experience_size == 0:
                self.ExperiencePool = {'s': hist_s,
                                       'a': hist_a,
                                       'r': hist_r,
                                       'next_s': hist_n_s,
                                       'terminal': hist_tmn}
            else:
                self.ExperiencePool['s'] = np.insert(self.ExperiencePool['s'], 0, values=hist_s, axis=0)  # (N, 10, 7)
                self.ExperiencePool['r'] = np.insert(self.ExperiencePool['r'], 0, values=hist_r, axis=0)
                self.ExperiencePool['a'] = np.insert(self.ExperiencePool['a'], 0, values=hist_a, axis=0)
                self.ExperiencePool['next_s'] = np.insert(self.ExperiencePool['next_s'], 0, values=hist_n_s, axis=0)
                self.ExperiencePool['terminal'] = np.insert(self.ExperiencePool['terminal'], 0, values=hist_tmn, axis=0)
            if np.max(hist_r) > self.max_reward:
                self.max_reward = np.max(hist_r)
            if np.max(hist_r) < self.min_reward:
                self.min_reward = np.min(hist_r)

            self.experience_size += 1

    def sample_index(self):
        index = np.random.randint(0, self.experience_size, size=self.batch_size)
        return index

    def fetch_sample_experience(self, index):
        o_states = np.array(self.ExperiencePool['s'])[index]  # (batch, 10, 7)
        o_actions = np.array(self.ExperiencePool['a'])[index]
        o_next_states = np.array(self.ExperiencePool['next_s'])[index]
        o_rewards = -1 * np.array(self.ExperiencePool['r'])[index] / self.min_reward
        o_done = np.array(self.ExperiencePool['terminal'])[index]
        if self.experience_size > self.max_experience_size:
            self._clip()
        return o_states, o_next_states, o_actions, o_rewards, o_done
