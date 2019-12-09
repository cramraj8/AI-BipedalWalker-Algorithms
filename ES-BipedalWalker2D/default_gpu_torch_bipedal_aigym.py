
import numpy as np
import time
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision import transforms
from pytorch_es import EvolutionModule
from torch.autograd import Variable

import gym


device = True if torch.cuda.is_available() else False

transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor()
])


class InvadersModel(nn.Module):
    def __init__(self, num_hl=25):
        super(InvadersModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(24, num_hl),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(num_hl, 4),
            # nn.ReLU(),
            nn.Tanh(),
        )

    def forward(self, input):
        main = self.main(input)
        return main


model = InvadersModel(num_hl=25)
if device:
    model = model.cuda()


def get_reward(weights, model, render=False):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data
    env = gym.make("BipedalWalker-v2")
    ob = env.reset()
    done = False
    total_reward = 0
    state = self.env.reset()

    while not done:
        if render:
            env.render()
        state = transform(state)
        if device:
            state = state.cuda()

        prediction = cloned_model(Variable(state, volatile=True))
        action = np.argmax(prediction.data)
        ob, reward, done, _ = env.step(action)
    total_reward += reward
    env.close()
    return total_reward


partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())
print("Starting ES Optim >>>")
es = EvolutionModule(
    mother_parameters, partial_func, population_size=50,
    sigma=0.3, learning_rate=0.001,
    reward_goal=200, consecutive_goal_stopping=20,
    threadcount=15, cuda=device, render_test=True
)
start = time.time()
final_weights = es.run(10000, print_step=10)
end = time.time() - start

weights_path = 'cpkpt_dir_atari'

pickle.dump(final_weights, open(os.path.abspath(weights_path), 'wb'))
final_reward = partial_func(final_weights, render=True)
# print(f'Reward from final weights: {final_reward}')
# print(f'Time to completion: {end}')
