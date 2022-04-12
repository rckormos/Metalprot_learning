"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains the architecture of the metalprot_learning model version 1. This model performs prediction of backbone distances to a metal.
"""

#imports 
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SingleLayerNet(nn.Module):
    def __init__(self):
        super(SingleLayerNet, self).__init__()
        self.flatten = nn.flatten()
        self.block1 = nn.sequential(
            nn.linear(12*12*16, 9) #check up on dimensions
            nn.ReLu(),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.block1(x)
        return y