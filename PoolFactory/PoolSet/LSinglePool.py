import numpy as np
from math import exp
from collections import deque
import random
from PoolFactory.Pool import ExperiencePool


class LSinglePool(ExperiencePool):
    def __init__(self, numb_agent, batch_size, max_size):
        ExperiencePool.__init__(self, numb_agent, batch_size, max_size)
        self._episode = []
        self.max_reward = 10
        self.min_reward = -10
        self._t = Temperature(4)
        self.ExperiencePool = deque([])

        # TODO Need to change
        self._tmc = 1  # Temperature Moderation Coefficient
        self._leniency_threshold = 200000  # Threshold

    def store(self, experience):
        _, _, action, _, terminal, _ = experience
        self._t.updateTemperatures([experience])
        temperature = self._t.getTemperatureUsingIndex(experience[5], action)
        leniency = 1 - exp(-self._tmc * temperature)
        experience.append(leniency)
        experience.append(temperature)
        self.ExperiencePool.append(experience)
        # if experience[4]:
        #     self._t.incEps()
        # if experience[3] >= 0.0 and experience[4] == 1 and self.aboveLeniencyThreshold():
        # if experience[3] >= 0.0 and experience[4] == 1:
        # Update temperatures for state action pairs visited
        # self._episode.append(experience)
        self.experience_size += 1
        # if experience[4]:  # If the transition is terminal
        #     self.experience_size += len(self._episode)
        #     while self.isFull():
        #         deletedEpisode = self.ExperiencePool.popleft()  # Pop first entry if RM is full
        #         self.experience_size -= len(deletedEpisode)
        #     self.ExperiencePool.append(self._episode)  # Store episode
        #     self._t.updateTemperatures(self._episode)
        #     self._episode = []  # Reset for next episode
        for i in range(len(experience)):
            if experience[3] > self.max_reward:
                self.max_reward = experience[3]
            if experience[3] < self.min_reward:
                self.min_reward = experience[3]

    def isFull(self):
        '''
        :return bool: True if RM is full, false otherwise
        '''
        return True if self.experience_size >= self.max_experience_size else False

    def sample_index(self):
        index = np.random.randint(0, self.experience_size, size=self.batch_size)
        return index

    def aboveLeniencyThreshold(self):
        """
        :return bool: True if the number of transitions stored
                       is above the learning threshold.
        """
        return True if self.experience_size > self._leniency_threshold else False

    def get_mini_batch(self):
        samples = []  # List used to store n traces used for sampling
        # Episodes are randomly choosen for sequence sampling:
        indexes = [random.randrange(len(self.ExperiencePool)) for i in range(self.batch_size)]
        # From each of the episodes a sequence is selected:
        s = []
        n_s = []
        a = []
        r = []
        d = []
        leniency = []
        for i in indexes:
            s.append(self.ExperiencePool[i][0])
            n_s.append(self.ExperiencePool[i][1])
            a.append(self.ExperiencePool[i][2])
            r.append(self.ExperiencePool[i][3] / abs(self.min_reward))
            d.append(self.ExperiencePool[i][4])
            leniency.append(self.ExperiencePool[i][6])
        s = np.squeeze(s)
        n_s = np.squeeze(n_s)
        a = np.squeeze(a)
        r = np.array(r).reshape(-1, 1)
        d = np.array(d).reshape(-1, 1)
        leniency = np.squeeze(leniency)
        return s, n_s, a, r, d, leniency

    def getHashKey(self, o_t):
        '''
        Loads hash-key for observation o_t
        :param tensor o_t: Observation for which key is required.
        :return int: hash key for observation o_t
        '''
        return self._t.getHash(o_t)


class Temperature:

    def __init__(self, nActions):
        '''
        :param dict config: Supplies hyperparameters
        :param int nActions: Number of actions used for indexing
        '''
        self._betas = []
        self._nActions = nActions
        self.temperatures = [dict() for i in range(nActions)]
        # max
        self._maxTemperature = 1
        # max_temperature_decay
        self._maxTemperatureDecay = 0.9998
        # exp_rho
        self.beta_0 = 0.9
        # len
        self.trace_len = 500
        # max decay
        self.max_decay = 1
        # tdf
        self.d = 0.99

        self.initTempDecayTrace()
        self.terminalHashkeys = []
        # self.eps = 0

    def getMaxTemperature(self):
        '''
        :return float: Max temperature
        '''
        return self._maxTemperature

    def initTempDecayTrace(self):
        '''
        Temperatuer Decay Schedule (TDS) initialisaion. Creates
        a schedule which can be used to retroactively decay temperature
        values, decaying temperatures near terminal near terminal states
        at a faster rate commpared to earlier transition.
        '''
        self._betas.append(self.beta_0)
        # for i in range(1, self.trace_len):
        for i in range(0, self.trace_len):
            t = exp(-2 * pow(self.beta_0, pow(self.d, i)))
            if t > self.max_decay:
                t = self.max_decay
            self._betas.append(t)

    def getHash(self, s_t):
        '''
        Get hash key for state s_t. For smaller
        discrete environments xxhash can be used,
        while larger environments require the
        grouping of semantically simmilar states.
        :param tensor s_t: state for which key needs to be obtained
        :return int: Hash key for state
        '''
        return hash(str(s_t))

    def getTemperature(self, o_t, action):
        """
        Get temperature for state action pair.
        :param tensor o_t: observation
        :param int action: action
        """
        index = self.getHash(o_t)
        return self.getTemperatureUsingIndex(index, action), index

    def getTemperatureUsingIndex(self, index, action):
        """
        Get temperature for state action pair using index key.
        :param int index: Hash key for a state
        :param int action: action
        """
        # If the index key already exists in the hash table:
        action = int(action)
        if index in self.temperatures[action]:
            # If temporature is less than the decaying max temperature:
            if self._maxTemperature > self.temperatures[action][index]:
                return self.temperatures[action][index]
            # Else return the max temperature:
            else:
                return self._maxTemperature
        else:
            # If undefined set temperature to current max temperature
            # self.temperatures[action][index] = self._maxTemperature
            return self._maxTemperature

    def getAvgTempUsingIndex(self, index):
        '''
        Calculates average temperature for a state based on index.
        :param int index: Hash key for a state.
        :return float: average temperature for the state belongin to index.
        '''
        temperatures = []
        for i in range(self._nActions):
            temperatures.append(self.getTemperatureUsingIndex(index, i))
        return sum(temperatures) / float(self._nActions)

    def getAvgTemp(self, observation):
        """
        Returns average temperature for a state.
        :param tensor observation: Observation for which the avg temperature is calcuted
        :return float: Avg temperature value for state
        """
        index = self.getHash(observation)
        return self.getAvgTempUsingIndex(index)

    def decayMaxTemperature(self):
        """
        Decays the max (global) temperature
        """
        self._maxTemperature *= self._maxTemperatureDecay

    def updateTemperatures(self, episode):
        """
        Decays temperatures for state actions pairs in episode.
        """
        decIndex = 0
        self.decayMaxTemperature()
        for transition in reversed(episode):
            s_t, s_tp1, action, _, terminal, idx = transition
            # if terminal:
            #     self.terminalHashkeys.append(idx)
            self.applyTDS(idx, action, decIndex)
            decIndex += 1

    def applyTDS(self, index, action, decIndex):
        """
         Decay temperature using TDS (Temperature Decay Schedule)
        :param int index: Hash key of state
        :param int action: Action used at time t
        :param int decIndex: Index of beta value to use for decay
        """
        # for action in range(self._nActions):
        action = int(action)
        if index in self.temperatures[action] and decIndex < self.trace_len:
            self.temperatures[action][index] *= self._betas[decIndex]
        else:
            self.temperatures[action][index] = self._maxTemperature
        if self._maxTemperature < self.temperatures[action][index]:
            self.temperatures[action][index] = self._maxTemperature

