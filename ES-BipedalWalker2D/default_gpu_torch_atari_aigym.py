
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
from PIL import Image

# cuda = args.cuda and torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = True if torch.cuda.is_available() else False

num_features = 16
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


class InvadersModel(nn.Module):
    def __init__(self, num_features):
        super(InvadersModel, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 8, num_features *
                      16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 16, 6, 4, 1, 0, bias=False),
            nn.Softmax(1)
        )

    def forward(self, input):
        main = self.main(input)
        return main


model = InvadersModel(num_features)
if device:
    model = model.cuda()


def get_reward(weights, model, render=False):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data
    env = gym.make("SpaceInvaders-v0")
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
        image = transform(Image.fromarray(ob))
        image = image.unsqueeze(0)
        if device:
            image = image.cuda()
        prediction = cloned_model(Variable(image, volatile=True))
        action = np.argmax(prediction.data)
        ob, reward, done, _ = env.step(action)
    total_reward += reward
    env.close()
    return total_reward


partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())
es = EvolutionModule(
    mother_parameters, partial_func, population_size=50,
    sigma=0.3, learning_rate=0.001,
    reward_goal=200, consecutive_goal_stopping=20,
    threadcount=15, cuda=device, render_test=True
)
start = time.time()
final_weights = es.run(2, print_step=10)
end = time.time() - start

weights_path = 'cpkpt_dir_atari'


pickle.dump(final_weights, open(os.path.abspath(weights_path), 'wb'))
final_reward = partial_func(final_weights, render=True)
print(format('Reward from final weights: {final_reward}'))
print(format('Time to completion: {end}'))

