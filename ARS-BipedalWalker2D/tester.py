

# Augmented Random Search

from pyvirtualdisplay import Display
import os
import numpy as np
import gym
from gym import wrappers

# Start virtual display
display = Display(visible=0, size=(1024, 768))
display.start()
os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)

class HP():
	# Hyperparameters
    def __init__(self,
				 nb_steps=1000,
				 episode_length=2000,
				 learning_rate=0.02,
				 num_deltas=16,
				 num_best_deltas=16,
				 noise=0.03,
				 seed=1,
				 env_name='BipedalWalker-v2',
				 record_every=50):

		self.nb_steps = nb_steps
		self.episode_length = episode_length
		self.learning_rate = learning_rate
		self.num_deltas = num_deltas
		self.num_best_deltas = num_best_deltas
		assert self.num_best_deltas <= self.num_deltas
		self.noise = noise
		self.seed = seed
		self.env_name = env_name
		self.record_every = record_every

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
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

class Policy():
    def __init__(self, input_size, output_size, hp, test_mode):
        self.theta = np.zeros((output_size, input_size))
        self.hp = hp
        self.test_mode = test_mode
        if self.test_mode: self.policy_model = np.load(MODEL_DIR)


    def evaluate(self, input, delta = None, direction = None):
        if (direction is None) and (self.test_mode):
            return self.policy_model.dot(input)
        elif direction is None:
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + self.hp.noise * delta).dot(input)
        elif direction == "-":
            return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]

    def update(self, rollouts, sigma_rewards):
        # sigma_rewards is the standard deviation of the rewards
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step

class ARSTester():
    def __init__(self,
                 hp=None,
                 input_size=None,
                 output_size=None,
                 normalizer=None,
                 policy=None,
                 monitor_dir=None,
                 do_test=False):

        self.hp = hp or HP()
        self.test_mode = do_test
        np.random.seed(self.hp.seed)
        self.env = gym.make(self.hp.env_name)
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.hp.episode_length = self.env.spec.timestep_limit or self.hp.episode_length
        self.input_size = input_size or self.env.observation_space.shape[0]
        self.output_size = output_size or self.env.action_space.shape[0]
        self.normalizer = normalizer or Normalizer(self.input_size)
        self.policy = policy or Policy(self.input_size, self.output_size, self.hp, self.test_mode)
        self.record_video = False

    # Explore the policy on one specific direction and over one episode
    def explore(self, direction=None, delta=None):
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewards = 0.0
        while not done and num_plays < self.hp.episode_length:
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state, delta, direction)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def test(self):
        done = False
        state = self.env.reset()
        sum_rewards = 0.0

        while not done:

            if done:
                break

            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)

            action = self.policy.evaluate(state, delta=None, direction=None)

            self.env.render()

            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)

            print('Step reward >>> ', reward)
            sum_rewards += reward

        print('Cumulative average sum of rewards >>> >>> ', sum_rewards)

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == '__main__':

    ENV_NAME = 'BipedalWalker-v2'
    MODEL_DIR = 'theta_table_values.npy'

    videos_dir = mkdir('.', 'videos-test')
    monitor_dir = mkdir(videos_dir, ENV_NAME)

    hp = HP(env_name=ENV_NAME)

    trainer = ARSTester(hp=hp, monitor_dir=monitor_dir, do_test=True)
    trainer.test()





