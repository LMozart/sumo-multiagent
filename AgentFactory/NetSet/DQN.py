from keras.initializers import he_uniform
from keras.layers import Dense, Input, concatenate, Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

# HIDDEN1_UNITS = 40
# HIDDEN2_UNITS = 60

HIDDEN1_UNITS = 70
HIDDEN2_UNITS = 90


class QNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.TAU = TAU
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        # Now create the model
        self.model, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_state = self.create_critic_network(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.sess.run(tf.global_variables_initializer())
        self.loss = 0

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def train_on_batch(self, y, x):
        a = self.model.fit(x=x, y=y, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)
        self.loss = a.history['loss'][0]

    def create_critic_network(self, state_size, action_size):
        S = Input(shape=(state_size,))
        h1 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(S)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(h1)
        h3 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(h2)
        h4 = Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer=he_uniform(seed=0))(h3)
        # h5 = Dense(action_size + 1, activation='linear')(h4)
        # V = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
        #            output_shape=(self.action_size,))(h5)
        V = Dense(action_size, activation='linear', kernel_initializer=he_uniform(seed=0))(h4)
        model = Model(inputs=S, outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, S
