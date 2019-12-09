

import gym
import random
import torch
import numpy as np
from collections import deque
import time
import os
import pandas as pd
from glob import glob

from ddpg_agent import Agent

# Setting user defined variables & constants
FREEZE_ACTOR = True
PRETRAINED_ACTOR_DIR = "../ars_theta_table_values.npy"
WEIGHTS_DIR = './ckpt_dir/'
actor_dir = os.path.join(WEIGHTS_DIR, 'ACTOR_NET')
critic_dir = os.path.join(WEIGHTS_DIR, 'CRITIC_NET')
max_savefiles = 10
record_interval = 100


env = gym.make('BipedalWalker-v2')
env.seed(10)

if not os.path.exists(actor_dir):
    os.makedirs(actor_dir)
if not os.path.exists(critic_dir):
    os.makedirs(critic_dir)


def is_dir_overflowing():
    num_files = len(os.listdir(actor_dir))
    if num_files > max_savefiles:
        return True
    else:
        return False


# agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10, freeze_actor=FREEZE_ACTOR, pretrained_actor=PRETRAINED_ACTOR_DIR)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0],
              random_seed=10, freeze_actor=False, pretrained_actor=None)


def ddpg(n_episodes=10000, max_t=700):
    start_time = time.time()

    track_avg_rewards = {}
    track_avg_rewards['Rewards'] = []

    track_elap_time = {}
    track_elap_time['Time'] = []
    track_elap_time['FormattertTime'] = []

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.step(state, action, reward, next_state, done)

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

        if i_episode % record_interval == 0:

            if is_dir_overflowing():

                # Removing oldest ACTOR weights file
                ckpt_actor_files = os.listdir(actor_dir)
                req_ckpt_actor_files = [
                    int(e.split('.')[-2].split('_')[-1]) for e in ckpt_actor_files]
                req_ckpt_actor_files.sort()
                os.remove(os.path.join(
                    actor_dir, "actor_weights_i_%s.pkl" % req_ckpt_actor_files[0]))
                # print(os.path.join(
                #     actor_dir, "actor_weights_i_%s.pkl" % req_ckpt_actor_files[0]))

                # Removing oldest CRITIC weights file
                ckpt_critic_files = os.listdir(critic_dir)
                req_ckpt_critic_files = [
                    int(e.split('.')[-2].split('_')[-1]) for e in ckpt_critic_files]
                req_ckpt_critic_files.sort()
                os.remove(os.path.join(
                    critic_dir, "critic_weights_i_%s.pkl" % req_ckpt_critic_files[0]))
                # print(os.path.join(
                #     actor_dir, "actor_weights_i_%s.pkl" % req_ckpt_critic_files[0]))

            # Saving WEGITHS file for both ACTOR + CRITIC
            torch.save(agent.actor_local.state_dict(), os.path.join(
                actor_dir, 'actor_weights_i_%s.pkl' % i_episode))
            torch.save(agent.critic_local.state_dict(), os.path.join(
                critic_dir, 'critic_weights_i_%s.pkl' % i_episode))

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


scores = ddpg()
