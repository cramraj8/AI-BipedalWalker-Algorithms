import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256, freeze=False, pretrained_actor=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, action_size)

        # Load either pretrained model or random initialize
        if pretrained_actor:
            self.fc1.weight = torch.nn.Parameter(
                torch.Tensor(np.load(pretrained_actor)))
        else:
            self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # self.fcs1 = nn.Linear(state_size + action_size, fcs1_units)
        # self.fc2 = nn.Linear(fcs1_units, 1)
        self.fcs1 = nn.Linear(state_size + action_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # xs = F.leaky_relu(self.fcs1(torch.cat((state, action), dim=1)))
        # x = F.leaky_relu(self.fc2(x))

        # xs = self.fcs1(torch.cat((state, action), dim=1))
        # x = F.leaky_relu(self.fc2(F.leaky_relu(xs)))

        return self.fcs1(torch.cat((state, action), dim=1))







