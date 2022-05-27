"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for Metalprot_learning models.
"""

#imports
import torch
from collections import OrderedDict

class SingleLayerNet(torch.nn.Module):
    def __init__(self, arch: list):
        super(SingleLayerNet, self).__init__()

        activation_function_dict = {'ReLU': torch.nn.ReLU()}

        layers = []
        for ind, layer in enumerate(arch):
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            layers.append((f'layer{ind}', torch.nn.Linear(input_dim, output_dim)))

            activation_key = layer['activation']
            activation = activation_function_dict[activation_key] if activation_key else None
            if activation:
                layers.append((f'activation{ind}', activation))

            dropout = torch.nn.Dropout(layer['dropout']) if layer['dropout'] else None
            if dropout:
                layers.append(f'droupout{ind}', dropout)

        self.block1 = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        y = self.block1(x.float())
        return y 

class SingleLayerNetV2(torch.nn.Module):
    def __init__(self, input_dim: int, l1: int, l2: int, output_dim: int):
        super(SingleLayerNetV2, self).__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, l1),
            torch.nn.ReLU(),
            torch.nn.Linear(l1,l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2,output_dim),
            torch.nn.ReLU()
        )

    def forward(self, x):
        y = self.block1(x.float())
        return y 

class DoubleLayerNet(torch.nn.Module):
    def __init__(self, input_dim: int, l1: int, l2: int, l3: int, output_dim: int, input_dropout: float, hidden_dropout: float):
        super(DoubleLayerNet, self).__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, l1),
            torch.nn.Dropout(input_dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(l1,l2),
            torch.nn.Dropout(hidden_dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(l2,l3),
            torch.nn.Dropout(hidden_dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(l3,output_dim),
            torch.nn.ReLU()
        )

    def forward(self, x):
        y = self.block1(x.float())
        return y 
