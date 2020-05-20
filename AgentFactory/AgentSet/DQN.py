from AgentFactory.Agent import Agent
from AgentFactory.NetSet.DQN import QNetwork
import numpy as np


class MLPLight(Agent):
    def __init__(self, agent_config, numb_a, action_size, state_size):
        Agent.__init__(self, agent_config, numb_a, action_size, state_size)
        lrc = agent_config.lrc
        print("THIS AGENT IS MLPLIGHT")
        self.TAU = agent_config.TAU
        self.eps = agent_config.eps
        self.min_eps = 0
        self.decay = 0.9
        self.action_size = action_size
        self.critic = QNetwork(self.sess,
                               TAU=self.TAU,
                               LEARNING_RATE=lrc,
                               BATCH_SIZE=self.n_batch,
                               action_size=action_size[0],
                               state_size=self.state_size[0])

    def target_q(self, state, index):
        return []

    def get_action(self, state, t):
        actions = []
        for i in range(self.n):
            # get newest weights before acting
            # get q values of current state
            if np.random.uniform(0.0, 1.0) < self.eps:
                # act randomly
                action = np.random.randint(self.action_size[0])
            else:
                # act greedily
                s = state[i].reshape(1, -1)
                q_state = self.critic.model.predict(s)
                action = np.argmax(q_state)
            actions.append(action)
            self.decay_eps()
        return actions

    def decay_eps(self):
        self.eps = max(self.eps * self.decay, self.min_eps)

    def load(self, path):
        self.critic.model.load_weights(path + "/{}-{}-{}.h5".format('actor', 0, 'DQN'))

    def save(self, path):
        self.critic.model.save(path + "/{}-{}-{}.h5".format('actor', 0, 'DQN'))
