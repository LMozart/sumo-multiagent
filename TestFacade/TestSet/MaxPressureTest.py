import numpy as np
from TestFacade.NoRLTest import NoRLTest


class PRESSURE(NoRLTest):
    def __init__(self, args):
        NoRLTest.__init__(self, args)

    def run(self):
        ep = 0
        average_rewards = []
        self.env.reset()
        for i in range(self.epoch):
            rewards, done = self.env.step([0])

            total_reward = []
            for r in rewards:
                total_reward.append(r)
            average_reward = np.average(total_reward)
            average_rewards.append(average_reward)

            if done:
                print('epoch:{} === rewards:{}'.format(str(ep), str(np.sum(average_rewards))))
                average_rewards = []
                self.env.reset()

    def load(self, path):
        pass
        # self.agents.actors.model.load_weights(path)
