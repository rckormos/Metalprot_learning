"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains the architecture of the metalprot_learning model version 1. This model performs prediction of backbone distances to a metal.
"""

#imports 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SingleLayerNet(nn.Module):
    def __init__(self, epochs: int, lr: float, batch_size: int, loss_fn: str, optimizer: str):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer

        loss_fn_dict = {'MAE': nn.L1Loss(),
                        'MSE': nn.MSELoss()}
        self.loss_fn = loss_fn_dict[loss_fn]

        super(SingleLayerNet, self).__init__()
        optimizer_dict = {'SGD': torch.optim.SGD(self.parameters(), lr=lr)}
        self.optimizer = optimizer_dict[optimizer]

        self.flatten = nn.flatten()
        self.block1 = nn.sequential(
            nn.linear(384, 9),
            nn.ReLu(),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.block1(x)
        return y

def train_loop(model, dataloader):

    for batch, (X, y) in enumerate(dataloader):
        
        #make prediction
        prediction = model.forward(X) 
        loss = model.loss_fn(prediction, y)

        #backpropagate and update weights
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    return loss.item()

def train_model(model, data):
    for epochs in range(0, model.epochs):
        pass
    pass