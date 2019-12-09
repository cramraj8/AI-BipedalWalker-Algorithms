

import gym
import random
import torch
import numpy as np
from collections import deque
import time

from ddpg_agent import Agent

FREEZE_ACTOR = True
PRETRAINED_ACTOR_DIR = "../ars_theta_table_values.npy"
WEIGHTS_DIR = './ckpt_dir/'

env = gym.make('BipedalWalker-v2')
env.seed(10)

if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)


# agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10, freeze_actor=FREEZE_ACTOR, pretrained_actor=PRETRAINED_ACTOR_DIR)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0],
              random_seed=10, freeze_actor=False, pretrained_actor=None)





def ddpg(n_episodes=2000, max_t=700):
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

        elapsed_time = time.time() - start_time
        formatted_elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(elapsed_time))

        track_elap_time['Time'].append(elapsed_time)
        track_elap_time['FormattertTime'].append(formatted_elapsed_time)

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tElapsed Time {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(
            i_episode, formatted_elapsed_time, np.mean(scores_deque), score), end="")
        


        if i_episode % 2 == 0:

            actor_dir = os.path.join(WEIGHTS_DIR, 'ACTOR_NET')
            critic_dir = os.path.join(WEIGHTS_DIR, 'CRITIC_NET')

            def is_dir_overflowing():
                num_files = len(os.listdir(actor_dir))
                if num_files > 5:
                    return True
                else:
                    return False

            if is_dir_overflowing():
                ckpt_files = glob(os.path.join(actor_dir, "checkpoint_actor_*.npy"))
                ckpt_files.sort()
                print('removing file : ', ckpt_files[0])
                os.remove(ckpt_files[0])

            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_%s.pth' % i_episode)
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_%s.pth' % i_episode)

            print('\rEpisode {}\tElapsed Time {}\tAverage Score: {:.2f}'.format(
                i_episode, formatted_elapsed_time,np.mean(scores_deque)))

        df_time = pd.DataFrame(track_elap_time)
        df_time.to_csv("elapsed_times.csv")

        track_avg_rewards['Rewards'].append(score)
        df = pd.DataFrame(track_avg_rewards)
        df.to_csv("scores.csv")

    return scores


scores = ddpg()
