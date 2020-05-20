import numpy as np
from keras.initializers import he_uniform
from keras.layers import Dense, Input
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import tensorflow.contrib as tc

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 60


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size,
                 BATCH_SIZE, TAU, LEARNING_RATE, seed):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        self.state_size = state_size
        self.seed = seed
        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)

        self.target_model.set_weights(self.model.get_weights())
        self.target_weights = self.weights
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        np.clip(np.array(action_grads), -5, 5, out=action_grads)
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        S = Input(shape=(state_size,))
        h0 = Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(h0)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(h1)
        V = Dense(action_dim, activation='tanh', kernel_initializer=he_uniform(seed=0))(h2)
        model = Model(inputs=S, outputs=V)
        return model, model.trainable_weights, S

    def get_action(self, state, t):
        # TODO Need to debug
        act_values = self.model.predict(np.array(state).reshape(-1, self.state_size))
        np.random.seed(t + self.seed)
        noise = np.random.uniform(-0.01, 0.01, act_values.shape[1])
        action_n = np.sum([act_values, noise], axis=0)
        np.clip(action_n, -1, 1, out=action_n)
        return action_n[0]  # returns action
