from keras.initializers import he_uniform
import tensorflow.contrib as tc
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 60


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, N, BATCH_SIZE, TAU, LEARNING_RATE):
        self.N = N
        self.TAU = TAU
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size,
                                                                         action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size,
                                                                                              action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.global_variables_initializer())
        self.loss = 0

    def gradients(self, states, actions):
        feed_dict = {}
        for i in range(self.N):
            feed_dict[self.state[i]] = states[i]
            feed_dict[self.action[i]] = actions[i]
        return self.sess.run(self.action_grads, feed_dict=feed_dict)

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def train_on_batch(self, y, x):
        a = self.model.fit(x=x, y=y, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)
        self.loss = a.history['loss'][0]

    def create_critic_network(self, state_size, action_dim):
        ActionInputs = []
        StateInputs = []
        for i in range(self.N):
            ActionInputs.append(Input(shape=(action_dim[i],)))
            StateInputs.append(Input(shape=(state_size[i],)))
        S = concatenate(StateInputs)
        A = concatenate(ActionInputs)
        a1 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(A)
        h2 = concatenate([h1, a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(h2)
        h4 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(h3)
        V = Dense(1, activation='linear', kernel_initializer=he_uniform(seed=0))(h4)
        model = Model(inputs=StateInputs + ActionInputs, outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, ActionInputs, StateInputs
