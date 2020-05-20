import numpy as np
from TestFacade.Test import Test
import tensorflow as tf


class MLPTest(Test):
    def __init__(self, args):
        Test.__init__(self, args)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.load("../Weight/DQN/2X2/51ep.h5")

    def run(self):
        ep = 0
        end = True
        average_rewards = []
        states = self.env.reset()
        for i in range(self.epoch):

            if end:
                actions = []
                for i in range(self.numb_a):
                    q_state = self.agents.critic.model.predict(states[i].reshape(1, -1))
                    actions.append(np.argmax(q_state))
            else:
                actions = [None] * self.numb_a
            next_states, rewards, done, end, end_index = self.env.step(actions)
            states = next_states
            total_reward = []
            for r in rewards:
                total_reward.append(r)
            average_reward = np.average(total_reward)
            average_rewards.append(average_reward)

            if done:
                print('epoch:{} === rewards:{}'.format(str(ep), str(np.sum(average_rewards))))
                average_rewards = []
                states = self.env.reset()

    def load(self, path):
        self.agents.critic.model.load_weights(path)
        print("Load Success")
