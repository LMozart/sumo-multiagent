import tensorflow as tf


class Agent:
    def __init__(self, agent_config, numb_a, action_size, state_size):
        self.n_batch = agent_config.batch
        self.state_size = state_size
        self.action_size = action_size
        self.n = numb_a

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=tf_config)

    def target_q(self, state, index):
        raise NotImplementedError()

    def get_action(self, state, t):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
