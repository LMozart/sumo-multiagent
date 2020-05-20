import os
import numpy as np
from TrainFacade.Trainer import Trainer
import tensorflow as tf


class LDQNTrainer(Trainer):
    def __init__(self, args):
        Trainer.__init__(self, args)
        with tf.variable_scope("DQN"):
            self.summary_writer = tf.summary.FileWriter(self.log_dir + '/' + str(0))
            self.critic_loss_vs = tf.Variable(0, dtype=tf.float32, name=str(0) + '_critic_loss_v')
            self.critic_loss_ops = tf.summary.scalar(str(0) + '_critic_loss_op', self.critic_loss_vs)

    def run(self):
        ep = 0
        end = True
        store = False
        end_index = []
        last_index = []
        last_states = None
        last_actions = None
        train_mark = False
        average_rewards = []
        ac_r = [0] * self.numb_a
        states = self.env.reset()
        # interval = 0

        for i in range(self.epoch):
            if end:
                actions = self.agents.get_action(states, i)
            else:
                actions = [None] * self.numb_a

            if i % 500 == 0:
                last_states = states
                last_actions = actions

            if end and i % 500 != 0:
                last_index = end_index
                store = True

            next_states, rewards, done, end, end_index = self.env.step(actions)
            # print(i, end)
            # collect_experience

            for j in range(self.numb_a):
                ac_r[j] += rewards[j]
            if store:
                for j in range(len(last_index)):
                    r = ac_r[last_index[j]]
                    index = self.pool.getHashKey(last_states[last_index[j]])
                    a = np.array(last_actions[last_index[j]]).reshape(-1, 1)
                    s = np.array(last_states[last_index[j]]).reshape(-1, self.env.input_numb[last_index[j]])
                    next_s = np.array(states[last_index[j]]).reshape(-1, self.env.input_numb[last_index[j]])
                    self.pool.store([s, next_s, a, r, done, index])
                    ac_r[last_index[j]] = 0
                last_states = states
                last_actions = actions
                store = False
            # if store:
            #     for j in range(len(last_index)):
            #         r = ac_r[last_index[j]]
            #         index = self.pool.getHashKey(states[last_index[j]])
            #         a = np.array(actions[last_index[j]]).reshape(-1, 1)
            #         s = np.array(states[last_index[j]]).reshape(-1, self.env.input_numb[last_index[j]])
            #         next_s = np.array(next_states[last_index[j]]).reshape(-1, self.env.input_numb[last_index[j]])
            #         self.pool.store([s, next_s, a, r, done, index])
            #         ac_r[last_index[j]] = 0
            #     store = False

            states = next_states
            total_reward = []
            for r in rewards:
                total_reward.append(r)
            average_reward = np.average(total_reward)
            average_rewards.append(average_reward)

            if self.pool.experience_size > self.learn_mark:
                self.train(i)
                train_mark = True
            if done:
                ep += 1
                end = True
                store = False
                self.save(self.model_dir, i)
                print("====================================")
                if train_mark:
                    print('[TRAIN]epoch:{} === rewards:{}'.format(str(ep), str(np.sum(average_rewards))))
                else:
                    print('[EXPLORE]epoch:{} === rewards:{}'.format(str(ep), str(np.sum(average_rewards))))
                self.reward_writer.add_summary(self.sess.run(self.tf_reward_op,
                                                            {self.tf_reward: np.sum(average_rewards)}), ep)
                average_rewards = []
                states = self.env.reset()

    def train(self, t):
        states, next_states, actions, rewards, done, leniency = self.pool.get_mini_batch()
        # batch forward q_s estimate
        Q_S = self.agents.critic.model.predict(states)
        # batch forward bootstrap
        rwd = self.next_state_bootstrap(next_states, done[:, 0], 0)
        q_targets = Q_S.copy()
        for k in range(self.n_batch):
            q_targets[k][actions[k]] = rewards[k] + self.gamma * rwd[k]
        # leniency = np.stack([leniency for i in range(4)]).T
        self.agents.critic.train_step_count = t
        self.agents.critic.train_on_batch(y=q_targets, x=states, leniency=leniency)
        self.agents.critic.target_train()

        if t % 100 == 0:
            self.summary_writer.add_summary(self.agents.sess.run(self.critic_loss_ops,
                                                                {self.critic_loss_vs:
                                                                 self.agents.critic.loss}), t)

    def next_state_bootstrap(self, next_states, terminals, index):
        q_next_s = self.agents.critic.target_model.predict(next_states)
        R = np.amax(q_next_s, axis=-1)
        return [0.0 if t is True else r for t, r in zip(terminals, R)]

    def get_actor_loss(self, next_states):
        target_action = self.agents.actors.target_model.predict(next_states)
        R = self.agents.critic.target_model.predict([next_states] + [target_action])
        return [r for r in R]

    def save(self, path, t):
        if not os.path.exists(path + '/' + str(t)):
            os.makedirs(path + '/' + str(t))
        self.agents.save(path + '/' + str(t))

    def load(self, path, t):
        raise NotImplementedError()
