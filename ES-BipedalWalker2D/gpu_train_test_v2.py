# Evolution Strategies BipedalWalker-v2
# https://blog.openai.com/evolution-strategies/
# gives good solution at around iter 100 in 5 minutes
# for testing model set reload=True

import gym
import numpy as np
import pickle as pickle
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is ",  device)


class HP():
    # Hyperparameters
    def __init__(self,
                 hl_size=25,  # hidden layer size
                 seed=10,
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
        self.n = torch.zeros(nb_inputs)
        self.mean = torch.zeros(nb_inputs)
        self.mean_diff = torch.zeros(nb_inputs)
        self.var = torch.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        # last_mean = self.mean.copy()
        last_mean = self.mean.clone()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = torch.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Policy(nn.Module):
    def __init__(self, input_size, output_size, hp):
        super(Policy, self).__init__()
        self.hp = hp
        if self.hp.test_mode:
            self.model = pickle.load(torch.load(
                open('model-pedal%d.pkl' % self.hp.version, 'rb')))
        else:
            self.model = {}
            self.model['W1'] = nn.Linear(24, self.hp.hl_size)
            # F.linear
            self.model['W2'] = nn.Linear(self.hp.hl_size, 4)
            self.reset_parameters()
            # self.model['W1'] = np.random.randn(
            #     24, self.hp.hl_size) / np.sqrt(24)
            # self.model['W2'] = np.random.randn(
            #     self.hp.hl_size, 4) / np.sqrt(self.hp.hl_size)

    def reset_parameters(self):
        self.model['W1'].weight.data.uniform_(*hidden_init(self.model['W1']))
        self.model['W2'].weight.data.uniform_(-3e-3, 3e-3)

    def evaluate(self, input_state):
        h1 = F.tanh(self.model['W1'](input_state))
        action = F.tanh(self.model['W2'](h1))
        return action


class ESTrainer(torch.nn.Module):
    def __init__(self,
                 hp=None,
                 input_size=None,
                 output_size=None,
                 normalizer=None,
                 policy=None,
                 ckpt_dir=None):
        super(ESTrainer, self).__init__()
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
            state = torch.tensor(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def train(self):
        for i in range(1001):
            N = {}
            # for k, v in self.policy.model.items():
            #     # N[k] = np.random.randn(self.hp.npop, v.shape[0], v.shape[1])
            #     # N[k] = np.random.randn(self.hp.npop, v.shape(0), v.shape(1))
            #     # N[k] = np.random.randn(self.hp.npop, v.size(0), v.size(1))
            #     print(v.size())
            # =========== Random initialize model parameters =========
            # k = 0
            # v = ['W1']; (24, self.hp.hl_size)
            # N[0] = np.random.randn(self.hp.npop, 24, self.hp.hl_size)
            N[0] = torch.zeros(self.hp.npop, 24, self.hp.hl_size)
            # k = 1
            # v = ['W2']; (self.hp.hl_size, 4)
            N[1] = torch.zeros(self.hp.npop, self.hp.hl_size, 4)


            R = torch.zeros(self.hp.npop)

            for j in range(self.hp.npop):
                model_try = {}
                # for k, v in self.policy.model.items():
                #     model_try[k] = v + self.hp.sigma*N[k][j]
                model_try[0] = v + self.hp.sigma*N[k][j]
                model_try[1] = v + self.hp.sigma*N[k][j]
                R[j] = self.explore(model_try)

            A = (R - np.mean(R)) / np.std(R)
            for k in self.policy.model:
                self.policy.model[k] = self.policy.model[k] + self.hp.alpha / \
                    (self.hp.npop*self.hp.sigma) * \
                    np.dot(N[k].transpose(1, 2, 0), A)

            cur_reward = self.explore(self.policy.model)
            self.hp.aver_reward = self.hp.aver_reward * 0.9 + cur_reward * \
                0.1 if self.hp.aver_reward != None else cur_reward
            print('iteration %d, current_reward: %.2f, average_reward: %.2f' %
                  (i, cur_reward, self.hp.aver_reward))

            if i % 5 == 0 and self.hp.allow_writing:
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
    trainer.to(device)
    trainer.train()
