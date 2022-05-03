"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for Metalprot_learning models and datasets.
"""

#imports
import numpy as np
import torch
from torch import nn
from collections import OrderedDict

class SingleLayerNet(nn.Module):
    def __init__(self, arch: list):
        super(SingleLayerNet, self).__init__()

        activation_function_dict = {'ReLU': nn.ReLU()}

        layers = []
        for ind, layer in enumerate(arch):
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            layers.append((f'layer{ind}', nn.Linear(input_dim, output_dim)))

            activation_key = layer['activation']
            activation = activation_function_dict[activation_key] if activation_key else None
            if activation:
                layers.append((f'activation{ind}', activation))

            dropout = nn.Dropout(layer['dropout']) if layer['dropout'] else None
            if dropout:
                layers.append((f'droupout{ind}', dropout))

        self.block1 = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        y = self.block1(x.float())
        return y

def SumOfSquaresLoss(output, target):
    loss = torch.sum(torch.square(output - target))
    return loss