from keras.initializers import he_uniform
from AgentFactory.Agent import Agent
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
from AgentFactory.NetSet.CriticNet import CriticNetwork
from AgentFactory.NetSet.ActorNet import ActorNetwork

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 60


class DDPG(Agent):
    def __init__(self, agent_config, numb_a, action_size, state_size):
        Agent.__init__(self, agent_config, numb_a, action_size, state_size)
        lrc = agent_config.lrc
        lra = agent_config.lra
        self.TAU = agent_config.TAU
        self.eps = agent_config.eps

        self.actors = MultiActorNetwork(self.sess,
                                        TAU=self.TAU,
                                        LEARNING_RATE=lra,
                                        BATCH_SIZE=self.n_batch,
                                        action_size=self.action_size[0],
                                        state_size=self.state_size[0],
                                        seed=0)
        self.critic = MultiCriticNetwork(self.sess,
                                         TAU=self.TAU,
                                         LEARNING_RATE=lrc,
                                         action_size=self.action_size[0],
                                         state_size=self.state_size[0],
                                         BATCH_SIZE=self.n_batch,
                                         N=self.n)

        # for i in range(self.n):
        #     actor = MultiActorNetwork(self.sess,
        #                               TAU=self.TAU,
        #                               LEARNING_RATE=lra,
        #                               BATCH_SIZE=self.n_batch,
        #                               action_size=self.action_size[i],
        #                               state_size=self.state_size[i],
        #                               seed=i)
        #     self.actors.append(actor)
        #     critic = MultiCriticNetwork(self.sess,
        #                                 TAU=self.TAU,
        #                                 LEARNING_RATE=lrc,
        #                                 action_size=self.action_size[i],
        #                                 state_size=self.state_size[i],
        #                                 BATCH_SIZE=self.n_batch,
        #                                 N=self.n)
        #     self.critic.append(critic)

    def target_q(self, state, index):
        target_action = self.actors.target_model.predict(state[index])
        return self.critic.target_model.predict([state[index]] + target_action)

    def get_action(self, state, t):
        action = []
        for i in range(self.n):
            action.append(self.actors.get_action(state[i], t))
        return action

    def load(self, path):
        self.actors.model.load_weights(path)
        self.critic.model.load_weights(path)

    def save(self, path):
        self.actors.model.save(path)
        self.critic.model.save(path)


class MultiCriticNetwork(CriticNetwork):
    def __init__(self, sess, state_size, action_size, N, BATCH_SIZE, TAU, LEARNING_RATE):
        super(MultiCriticNetwork, self).__init__(sess, state_size, action_size, N, BATCH_SIZE, TAU, LEARNING_RATE)

    def gradients(self, states, actions):
        feed_dict = {self.state: states, self.action: actions}
        return self.sess.run(self.action_grads, feed_dict=feed_dict)

    def create_critic_network(self, state_size, action_dim):
        A = Input(shape=(action_dim,))
        S = Input(shape=(state_size,))
        w1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear', kernel_initializer=he_uniform(seed=0))(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear', kernel_initializer=he_uniform(seed=0))(w1)
        h2 = concatenate([h1, a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(h2)
        h4 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(h3)
        V = Dense(1, activation='linear', kernel_initializer=he_uniform(seed=0))(h4)
        model = Model(inputs=[S] + [A], outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


class MultiActorNetwork(ActorNetwork):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, seed):
        super(MultiActorNetwork, self).__init__(sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, seed)
