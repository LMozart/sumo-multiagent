import numpy as np


class ExperiencePool:
    def __init__(self, numb_agent, batch_size, max_size):
        self.experience_size = 0
        self.numb_agent = numb_agent
        self.batch_size = batch_size
        self.max_experience_size = max_size
        self.ExperiencePool = [[] for i in range(self.numb_agent)]

    def _clip(self):
        for Pool in self.ExperiencePool:
            np.delete(Pool, -1, axis=0)
        self.experience_size -= 1

    def store(self, experiences):
        raise NotImplementedError()

    def sample_index(self):
        raise NotImplementedError()

    def fetch_sample_experience(self, index):
        raise NotImplementedError()
