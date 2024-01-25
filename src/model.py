"""Module containing model relevant functions
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import dvc.api


# Get parameter
PARAM_DICT = dvc.api.params_show()


# Q-Network
class DQN(nn.Module):
    """Class for Deep Q Network"""

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, input):
        """Forward funtion of model"""
        h_0 = F.relu(self.layer1(input))
        h_1 = F.relu(self.layer2(h_0))
        output = self.layer3(h_1)
        return output


