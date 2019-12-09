# Evolution Strategies BipedalWalker-v2
# https://blog.openai.com/evolution-strategies/
# gives good solution at around iter 100 in 5 minutes
# for testing model set reload=True

import gym
import numpy as np
import pickle as pickle
import sys

import dask
import psutil
# import dask.distributed

num_workers = 10
num_threads_per_worker = 1
scheduler = ''

"""
- Set n_pop
- Set num_workers

"""


if __name__ == '__main__':

    from dask.distributed import Client

    if not scheduler:
        if num_workers <= 0:
            num_workers_viable = max(
                1, psutil.cpu_count(logical=False) + num_workers)
        else:
            num_workers_viable = num_workers
        num_threads_per_worker_viable = (
            num_threads_per_worker if num_threads_per_worker >= 1 else None)

        print('Creating dask LocalCluster with %d worker(s), %d thread(s) per '
              'worker' % (num_workers_viable, num_threads_per_worker_viable))
        scheduler = dask.distributed.LocalCluster(
            ip='0.0.0.0',  # Allow reaching the diagnostics port externally
            scheduler_port=0,  # Don't expose the scheduler port
            n_workers=num_workers_viable,
            memory_limit=0,
            threads_per_worker=num_threads_per_worker_viable,
            silence_logs=False
        )
    else:
        scheduler = dask.distributed.LocalCluster(
            ip='0.0.0.0',  # Allow reaching the diagnostics port externally
            scheduler_port=0,  # Don't expose the scheduler port
            n_workers=num_workers,
            memory_limit=0,
            threads_per_worker=num_threads_per_worker,
            silence_logs=False
        )

    print(Client(scheduler))

    env = gym.make('BipedalWalker-v2')
    np.random.seed(10)
    hl_size = 25 # 100
    version = 1
    npop = 50  # 5
    sigma = 0.1
    alpha = 0.03
    iter_num = 300
    aver_reward = None
    allow_writing = True
    reload = False

    print(hl_size, version, npop, sigma, alpha, iter_num)

    if reload:
        model = pickle.load(open('model-pedal%d.pkl' % version, 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(24, hl_size) / np.sqrt(24)
        model['W2'] = np.random.randn(hl_size, 4) / np.sqrt(hl_size)

    def get_action(state, model):
        hl = np.matmul(state, model['W1'])
        hl = np.tanh(hl)
        action = np.matmul(hl, model['W2'])
        action = np.tanh(action)
        return action

    def f(model, render=False):
        state = env.reset()
        total_reward = 0
        for t in range(iter_num):
            if render:
                env.render()
            action = get_action(state, model)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    # testing
    if reload:
        iter_num = 10000
        print("================== TESTING ===================")
        for i_episode in range(1000):
            current_reward = f(model, True)
            aver_reward = aver_reward * 0.9 + current_reward * \
                0.1 if aver_reward != None else current_reward
            print('Episode %d, current_reward: %.2f, average_reward: %.2f' %
                  (i, current_reward, aver_reward))
        sys.exit('demo finished')

    for i in range(10001):
        N = {}
        for k, v in model.items():
            N[k] = np.random.randn(npop, v.shape[0], v.shape[1])
        # R = np.zeros(npop)

        temp_R = []
        for j in range(npop):
            model_try = {}
            for k, v in model.items():
                model_try[k] = v + sigma*N[k][j]
            temp_R.append(dask.delayed(f)(model_try))

        R = dask.compute(temp_R)[0]
        # print(R)
        # print(type(R))
        # print(R[0])

        A = (R - np.mean(R)) / np.std(R)
        for k in model:
            model[k] = model[k] + alpha / \
                (npop*sigma) * np.dot(N[k].transpose(1, 2, 0), A)

        cur_reward = f(model)
        aver_reward = aver_reward * 0.9 + cur_reward * \
            0.1 if aver_reward != None else cur_reward
        print('iter %d, cur_reward %.2f, aver_reward %.2f' %
              (i, cur_reward, aver_reward))

        if i % 10 == 0 and allow_writing:
            pickle.dump(model, open('model-pedal%d-dask.pkl' % version, 'wb'))