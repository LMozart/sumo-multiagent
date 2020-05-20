import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np


class TFACNet():
    def __init__(self, lra, lrc, action_size, state_size, index):
        self.layer_norm = False
        self.index = index
        with tf.device("/gpu:0"):
            self.a_in = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
            self.s_in = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            self.critic = self._build_critic_network(name=str(index) + '_online_critic')
            self.t_critic = self._build_critic_network(name=str(index) + '_target_critic')
            self.actor = self._build_actor_network(name=str(index) + '_online_actor',
                                                   action_dim=action_size)
            self.t_actor = self._build_actor_network(name=str(index) + '_target_actor',
                                                     action_dim=action_size)

            self.actor_optimizer = tf.train.AdamOptimizer(lra)
            self.critic_optimizer = tf.train.AdamOptimizer(lrc)

            self.actor_loss = -tf.reduce_mean(self._build_critic_network(name=str(index) + '_online_critic', reuse=True))
            self.actor_train = self.actor_optimizer.minimize(self.actor_loss)
            self.target_q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.critic_loss = tf.reduce_mean(tf.square(self.target_q - self.critic))
            self.critic_train = self.critic_optimizer.minimize(self.critic_loss)

        self.a_loss_value = 0
        self.c_loss_value = 0

    def _build_actor_network(self, name, action_dim):
        with tf.variable_scope(name) as scope:
            x = self.s_in
            x = tf.layers.dense(x, 256, activation=tf.nn.relu,
                                kernel_initializer=tf.keras.initializers.he_normal(0))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.dense(x, 256, activation=tf.nn.relu,
                                kernel_initializer=tf.keras.initializers.he_normal(0))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu,
                                kernel_initializer=tf.keras.initializers.he_normal(0))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.dense(x, action_dim, activation=tf.nn.tanh,
                                kernel_initializer=tf.keras.initializers.he_normal(0))

            return x

    def _build_critic_network(self, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            x = self.s_in
            x = tf.layers.dense(x, 256, activation=tf.nn.relu,
                                kernel_initializer=tf.keras.initializers.he_normal(0))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.concat([x, self.a_in], axis=-1)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.dense(x, 256, activation=tf.nn.relu,
                                kernel_initializer=tf.keras.initializers.he_normal(0))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu,
                                kernel_initializer=tf.keras.initializers.he_normal(0))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.keras.initializers.he_normal(0))
            return x

    def train_actor(self, state, action, sess, t):
        sess.run(self.actor_train, {self.s_in: state, self.a_in: action})
        if t % 50 == 0:
            self.a_loss_value = sess.run(self.actor_loss, {self.s_in: state, self.a_in: action})

    def train_critic(self, state, action, target, sess, t):
        sess.run(self.critic_train,
                 {self.s_in: state,
                  self.a_in: action,
                  self.target_q: target})
        if t % 50 == 0:
            self.c_loss_value = sess.run(self.critic_loss,
                                         {self.s_in: state,
                                          self.a_in: action,
                                          self.target_q: target})

    def action(self, state, sess):
        action = sess.run(self.actor, {self.s_in: state})
        noise = np.random.uniform(-0.01, 0.01, action.shape[1])
        action_n = np.sum([action, noise], axis=0)
        return action_n

    def Q(self, state, sess):
        action = sess.run(self.t_actor,
                          {self.s_in: state})
        return sess.run(self.t_critic,
                        {self.s_in: state,
                         self.a_in: action})