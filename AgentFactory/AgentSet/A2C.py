from AgentFactory.Agent import Agent
from AgentFactory.NetSet.DQN import QNetwork
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam


class A2C(Agent):
    def __init__(self, agent_config, numb_a, action_size, state_size):
        Agent.__init__(self, agent_config, numb_a, action_size, state_size)
        lra = agent_config.lra
        lrc = agent_config.lrc
        print("THIS AGENT IS MLPLIGHT")
        self.act_dim = action_size[0]
        self.env_dim = state_size
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(action_size[0], self.shared, lra)
        self.critic = Critic(self.env_dim[0], action_size[0], self.shared, lrc)
        # Build optimizers

        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()
        self.TAU = agent_config.TAU
        self.eps = agent_config.eps
        self.min_eps = 0
        self.decay = 0.9

    def train(self, gamma, rewards, actions, states):
        discounted_rewards = self.discount(rewards, gamma)
        state_values = self.critic.predict(np.array(states))
        advantages = np.squeeze(discounted_rewards - state_values)
        self.a_opt([np.array(states), np.array(actions), advantages])
        self.c_opt([np.array(states), discounted_rewards])

    def discount(self, r, gamma):
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input(self.env_dim,)
        x = Dense(40, activation='relu')(inp)
        x = Dense(60, activation='relu')(x)
        return Model(inp, x)

    def target_q(self, state, index):
        return []

    def get_action(self, state, t):
        actions = []
        for i in range(self.n):
            # get newest weights before acting
            # get q values of current state
            s = state[i].reshape(1, -1)
            action = np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]
            actions.append(action)
        return actions

    def decay_eps(self):
        self.eps = max(self.eps * self.decay, self.min_eps)

    def load(self, path):
        self.actor.model.load_weights(path + "/{}-{}-{}.h5".format('actor', 0, 'DQN'))

    def save(self, path):
        self.actor.model.save(path + "/{}-{}-{}.h5".format('actor', 0, 'DQN'))


class Critic:
    """ Critic for the A2C Algorithm
    """
    def __init__(self, inp_dim, out_dim, network, lr):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.model = self.addHead(network)
        self.discounted_r = K.placeholder(shape=(None,))
        self.adam_optimizer = Adam(lr=lr)

    def addHead(self, network):
        """ Assemble Critic network to predict value of each state
        """
        x = Dense(60, activation='relu')(network.output)
        out = Dense(1, activation='linear')(x)
        return Model(network.input, out)

    def optimizer(self):
        """ Critic Optimization: Mean Squared Error over discounted rewards
        """
        critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        updates = self.adam_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([self.model.input, self.discounted_r], [], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

    def predict(self, inp):
        """ Critic Value Prediction
        """
        return self.model.predict(inp)


class Actor:
    """ Actor for the A2C Algorithm
    """

    def __init__(self, out_dim, network, lr):
        self.out_dim = out_dim
        self.model = self.addHead(network)
        self.action_pl = K.placeholder(shape=(None, out_dim))
        self.advantages_pl = K.placeholder(shape=(None,))
        self.adam_optimizer = Adam(lr=lr)

    def addHead(self, network):
        """ Assemble Actor network to predict probability of each action
        """
        x = Dense(60, activation='relu')(network.output)
        out = Dense(self.out_dim, activation='softmax')(x)
        return Model(network.input, out)

    def optimizer(self):
        """ Actor Optimization: Advantages + Entropy term to encourage exploration
        (Cf. https://arxiv.org/abs/1602.01783)
        """
        weighted_actions = K.sum(self.action_pl * self.model.output, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        loss = 0.001 * entropy - K.sum(eligibility)

        updates = self.adam_optimizer.get_updates(self.model.trainable_weights, [], loss)
        return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

    def predict(self, inp):
        """ Critic Value Prediction
        """
        return self.model.predict(inp)
