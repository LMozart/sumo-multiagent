import os
import numpy as np
from TrainFacade.Trainer import Trainer
import tensorflow as tf


class DdpgTrainer(Trainer):
    def __init__(self, args):
        Trainer.__init__(self, args)
        with tf.variable_scope("ddpg"):
            self.summary_writer = tf.summary.FileWriter(self.log_dir + '/' + str(0))
            self.critic_loss_vs = tf.Variable(0, dtype=tf.float32, name=str(0) + '_critic_loss_v')
            self.critic_loss_ops = tf.summary.scalar(str(0) + '_critic_loss_op', self.critic_loss_vs)
            self.actor_loss_vs = tf.Variable(0, dtype=tf.float32, name=str(0) + '_actor_loss_v')
            self.actor_loss_ops = tf.summary.scalar(str(0) + '_actor_loss_op', self.actor_loss_vs)

    def run(self):
        ep = 0
        end = True
        store = False
        end_index = []
        last_index = []
        train_mark = False
        average_rewards = []
        last_states = None
        last_actions = None
        states = self.env.reset()
        ac_r = [0] * self.numb_a

        for i in range(self.epoch):
            # get action only at the start of the phase
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
            for j in range(self.numb_a):
                ac_r[j] += rewards[j]
            if store:
                experiences = [{}] * len(last_index)
                for j in range(len(experiences)):
                    experiences[j]['r'] = np.array(ac_r[last_index[j]]).reshape(1, 1)
                    ac_r[last_index[j]] = 0
                    experiences[j]['a'] = np.array(last_actions[last_index[j]]).reshape(-1, 1)
                    experiences[j]['s'] = np.array(last_states[last_index[j]]).reshape(-1, self.env.input_numb[last_index[j]])
                    experiences[j]['terminal'] = np.array(done).reshape(-1, 1)
                    experiences[j]['next_s'] = np.array(states[last_index[j]]).reshape(-1, self.env.input_numb[last_index[j]])
                self.pool.store(experiences)
                last_states = states
                last_actions = actions
                store = False
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
        index = self.pool.sample_index()
        states, next_states, actions, rewards, done = self.pool.fetch_sample_experience(index)

        q_targets = rewards.reshape(self.n_batch, -1)
        # q_targets = rewards
        # rwd = self.next_state_bootstrap(next_states, done[:, 0], 0)
        rwd = self.next_state_bootstrap(next_states, done[:, 0], 0)

        for l in range(self.n_batch):
            q_targets[l, 0] += self.gamma * rwd[l]

        self.agents.critic.train_on_batch(y=q_targets, x=[states] + [actions])
        grads = self.agents.critic.gradients(states=states,
                                             actions=self.agents.actors.model.predict(states))
        self.agents.actors.train(states=states,
                                 action_grads=grads[0])
        self.agents.actors.target_train()
        self.agents.critic.target_train()
        # self.actor_loss = np.average(self.get_actor_loss(max_s))
        self.actor_loss = np.average(self.get_actor_loss(states))
        if t % 100 == 0:
            self.summary_writer.add_summary(self.agents.sess.run(self.critic_loss_ops,
                                                                 {self.critic_loss_vs:
                                                                  self.agents.critic.loss}), t)
            self.summary_writer.add_summary(self.agents.sess.run(self.actor_loss_ops,
                                                                 {self.actor_loss_vs:
                                                                  self.actor_loss}), t)

    def next_state_bootstrap(self, next_states, terminals, index):
        target_action = self.agents.actors.target_model.predict(next_states)
        R = self.agents.critic.target_model.predict([next_states] + [target_action])
        return [0.0 if t is True else self.gamma * r for t, r in zip(terminals, R)]

    def get_actor_loss(self, next_states):
        target_action = self.agents.actors.target_model.predict(next_states)
        R = self.agents.critic.target_model.predict([next_states] + [target_action])
        return [r for r in R]

    def save(self, path, t):
        if not os.path.exists(path + '/' + str(t)):
            os.makedirs(path + '/' + str(t))
        for i in range(self.numb_a):
            self.agents.save(path + '/' + str(t) + "/{}-{}-{}-{}.h5".format('actor', i, 'ddpg', t))

    def load(self, path, t):
        raise NotImplementedError()
