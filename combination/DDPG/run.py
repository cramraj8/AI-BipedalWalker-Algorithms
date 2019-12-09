

import gym
import random
import torch
import numpy as np
from collections import deque

from ddpg_agent import Agent

FREEZE_ACTOR = True
# PRETRAINED_ACTOR_DIR = "../theta_table_values-3.npy"
PRETRAINED_ACTOR_DIR = "../ars_theta_table_values.npy"

env = gym.make('BipedalWalker-v2')
env.seed(10)
# agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10, freeze_actor=FREEZE_ACTOR, pretrained_actor=PRETRAINED_ACTOR_DIR)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10, freeze_actor=False, pretrained_actor=None)


def ddpg(n_episodes=2000, max_t=700):
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
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 5 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
    return scores

scores = ddpg()
