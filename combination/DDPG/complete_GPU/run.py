

import gym
import random
import torch
import numpy as np
from collections import deque
import time
import os
import pandas as pd
from glob import glob

import torch.nn as nn
import torchvision.transforms as transforms
from ddpg_agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Run(nn.Module):

    def __init__(self, freeze_actor, pretrained_actor_dir, weights_dir, max_savefiles, record_interval, n_episodes, max_t):
        super(Run, self).__init__()
        # agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10,
        # freeze_actor=FREEZE_ACTOR,
        # pretrained_actor=PRETRAINED_ACTOR_DIR)

        self.freeze_actor = freeze_actor
        self.pretrained_actor_dir = pretrained_actor_dir
        self.weights_dir = weights_dir
        self.max_savefiles = max_savefiles
        self.record_interval = record_interval

        self.n_episodes = n_episodes
        self.max_t = max_t

        self.actor_dir = os.path.join(self.weights_dir, 'ACTOR_NET')
        self.critic_dir = os.path.join(self.weights_dir, 'CRITIC_NET')
        if not os.path.exists(self.actor_dir):
            os.makedirs(self.actor_dir)
        if not os.path.exists(self.critic_dir):
            os.makedirs(self.critic_dir)

        self.env = gym.make('BipedalWalker-v2').unwrapped
        self.env.seed(10)

        self.agent = Agent(state_size=self.env.observation_space.shape[0], action_size=self.env.action_space.shape[0],
                           random_seed=10, freeze_actor=False, pretrained_actor=None).to(device)

        # self.transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])

    def is_dir_overflowing(self):
        num_files = len(os.listdir(self.actor_dir))
        if num_files > self.max_savefiles:
            return True
        else:
            return False

    def ddpg_train(self):
        start_time = time.time()

        track_avg_rewards = {}
        track_avg_rewards['Rewards'] = []

        track_elap_time = {}
        track_elap_time['Time'] = []
        track_elap_time['FormattertTime'] = []

        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            self.agent.reset()
            score = 0
            for t in range(self.max_t):

                # state_tensor = self.transform(state)
                state_tensor = torch.Tensor(state)
                if torch.cuda.is_available():
                    state_tensor = state_tensor.cuda()

                action = self.agent.act(state_tensor.cpu().data.numpy())
                next_state, reward, done, _ = self.env.step(action)

                self.agent.step(state, action, reward, next_state, done)

                state = next_state
                score += reward
                if done:
                    break

                # env.render()

            # Computing the formatted elapsed time
            elapsed_time = time.time() - start_time
            formatted_elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(elapsed_time))
            track_elap_time['Time'].append(elapsed_time)
            track_elap_time['FormattertTime'].append(formatted_elapsed_time)

            scores_deque.append(score)
            scores.append(score)
            print('\rEpisode {}\tElapsed Time {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(
                i_episode, formatted_elapsed_time, np.mean(scores_deque), score), end="")

            if i_episode % self.record_interval == 0:

                if self.is_dir_overflowing():

                    # Removing oldest ACTOR weights file
                    ckpt_actor_files = os.listdir(self.actor_dir)
                    req_ckpt_actor_files = [
                        int(e.split('.')[-2].split('_')[-1]) for e in ckpt_actor_files]
                    req_ckpt_actor_files.sort()
                    os.remove(os.path.join(
                        self.actor_dir, "actor_weights_i_%s.pkl" % req_ckpt_actor_files[0]))

                    # Removing oldest CRITIC weights file
                    ckpt_critic_files = os.listdir(self.critic_dir)
                    req_ckpt_critic_files = [
                        int(e.split('.')[-2].split('_')[-1]) for e in ckpt_critic_files]
                    req_ckpt_critic_files.sort()
                    os.remove(os.path.join(
                        self.critic_dir, "critic_weights_i_%s.pkl" % req_ckpt_critic_files[0]))

                # Saving WEGITHS file for both ACTOR + CRITIC
                torch.save(self.agent.actor_local.state_dict(), os.path.join(
                    self.actor_dir, 'actor_weights_i_%s.pkl' % i_episode))
                torch.save(self.agent.critic_local.state_dict(), os.path.join(
                    self.critic_dir, 'critic_weights_i_%s.pkl' % i_episode))

                # Printing average score for record INTERVALS
                print('\rEpisode {}\tElapsed Time {}\tAverage Score: {:.2f}'.format(
                    i_episode, formatted_elapsed_time, np.mean(scores_deque)))

            # Saving TIME csv
            df_time = pd.DataFrame(track_elap_time)
            df_time.to_csv("elapsed_times.csv")

            # Saving REWARDS csv
            track_avg_rewards['Rewards'].append(score)
            df = pd.DataFrame(track_avg_rewards)
            df.to_csv("scores.csv")

        return scores


if __name__ == '__main__':
    # Setting user defined variables & constants
    FREEZE_ACTOR = True
    PRETRAINED_ACTOR_DIR = "../ars_theta_table_values.npy"
    WEIGHTS_DIR = './ckpt_dir/'
    actor_dir = os.path.join(WEIGHTS_DIR, 'ACTOR_NET')
    critic_dir = os.path.join(WEIGHTS_DIR, 'CRITIC_NET')
    max_savefiles = 10
    record_interval = 100

    n_episodes = 10000
    max_t = 700

    run = Run(freeze_actor=FREEZE_ACTOR,
              pretrained_actor_dir=PRETRAINED_ACTOR_DIR,
              weights_dir=WEIGHTS_DIR,
              max_savefiles=max_savefiles,
              record_interval=record_interval,
              n_episodes=n_episodes,
              max_t=max_t)
    run.ddpg_train()
