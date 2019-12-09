# Evolution Strategies BipedalWalker-v2
# https://blog.openai.com/evolution-strategies/
# gives good solution at around iter 100 in 5 minutes
# for testing model set reload=True

import gym
import numpy as np
import pickle as pickle
import sys
import os


class HP():
    # Hyperparameters
    def __init__(self,
                 hl_size=50,  # hidden layer size
                 seed=44,
                 episode_length=1000,
                 nb_steps=1000,

                 npop=50,
                 sigma=0.1,
                 alpha=0.03,
                 iter_num=300,
                 aver_reward=None,
                 allow_writing=True,
                 reload=False,
                 env_name='BipedalWalker-v2',
                 version=1,
                 ):
        self.hl_size = hl_size
        self.nb_steps = nb_steps
        self.version = version
        self.npop = npop
        self.sigma = sigma
        self.alpha = alpha
        self.iter_num = iter_num
        self.aver_reward = aver_reward
        self.allow_writing = allow_writing
        self.test_mode = reload
        self.env_name = env_name
        self.seed = seed
        self.episode_length = episode_length


class Normalizer():
    # Normalizes the inputs
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class Policy():
    def __init__(self, input_size, output_size, hp):
        self.hp = hp
        if self.hp.test_mode:
            self.model = pickle.load(
                open('model-pedal%d.pkl' % self.hp.version, 'rb'))
        else:
            self.model = {}
            self.model['W1'] = np.random.randn(
                24, self.hp.hl_size) / np.sqrt(24)
            self.model['W2'] = np.random.randn(
                self.hp.hl_size, 4) / np.sqrt(self.hp.hl_size)

    def evaluate(self, input_state):
        hl = np.matmul(input_state, self.model['W1'])
        hl = np.tanh(hl)
        action = np.matmul(hl, self.model['W2'])
        action = np.tanh(action)
        return action


class ESTrainer():
    def __init__(self,
                 hp=None,
                 input_size=None,
                 output_size=None,
                 normalizer=None,
                 policy=None,
                 ckpt_dir=None):
        self.hp = hp or HP()
        np.random.seed(self.hp.seed)
        self.env = gym.make(self.hp.env_name)

        self.input_size = input_size or self.env.observation_space.shape[0]
        self.output_size = output_size or self.env.action_space.shape[0]

        self.normalizer = normalizer or Normalizer(self.input_size)
        self.policy = policy or Policy(
            self.input_size, self.output_size, self.hp)
        self.ckpt_dir = ckpt_dir

    # Explore the policy on one specific direction and over one episode
    def explore(self, direction=None, delta=None):
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewards = 0.0
        while not done and num_plays < self.hp.episode_length:
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def train(self):
        for i in range(10001):
            N = {}
            for k, v in self.policy.model.items():
                N[k] = np.random.randn(self.hp.npop, v.shape[0], v.shape[1])
            R = np.zeros(self.hp.npop)

            for j in range(self.hp.npop):
                model_try = {}
                for k, v in self.policy.model.items():
                    model_try[k] = v + self.hp.sigma*N[k][j]
                R[j] = self.explore(model_try)

            A = (R - np.mean(R)) / np.std(R)
            for k in self.policy.model:
                self.policy.model[k] = self.policy.model[k] + self.hp.alpha / \
                    (self.hp.npop*self.hp.sigma) * \
                    np.dot(N[k].transpose(1, 2, 0), A)

            cur_reward = self.explore(self.policy.model)
            aver_reward = None
            aver_reward = aver_reward * 0.9 + cur_reward * \
                0.1 if aver_reward != None else cur_reward
            print('iteration %d, current_reward: %.2f, average_reward: %.2f' %
                  (i, cur_reward, aver_reward))

            if i % 10 == 0 and self.hp.allow_writing:
                pickle.dump(self.policy.model, open(os.path.join(self.ckpt_dir,
                    'model-pedal%d.pkl' % self.hp.version), 'wb'))


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    ENV_NAME = 'BipedalWalker-v2'
    CKPT_DIR = mkdir('.', 'ckpt-train')
    hp = HP(env_name=ENV_NAME)

    trainer = ESTrainer(hp=hp, ckpt_dir=CKPT_DIR)
    trainer.train()
