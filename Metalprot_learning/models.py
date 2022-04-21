"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains the architecture of the metalprot_learning model version 1. This model performs prediction of backbone distances to a metal.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SingleLayerNet(nn.Module):
    def __init__(self, epochs: int, lr: float, batch_size: int, loss_fn: str, optimizer: str):
        super(SingleLayerNet, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer

        loss_fn_dict = {'MAE': nn.L1Loss(),
                        'MSE': nn.MSELoss()}
        self.loss_fn = loss_fn_dict[loss_fn]

        self.block1 = nn.Sequential(
            nn.Linear(384, 9),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.block1(x)
        return y

def train_loop(model, train_dataloader, optimizer):

    for batch, (X, y) in enumerate(train_dataloader):
        
        #make prediction
        prediction = model.forward(X) 
        loss = model.loss_fn(prediction, y)

        #backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def validation_loop(model, validation_dataloader):
    validation_loss = 0
    with torch.no_grad():
        for X,y in validation_dataloader:
            prediction = model(X)
            validation_loss += model.loss_fn(prediction,y).item()

    validation_loss /= len(validation_dataloader)
    return validation_loss

def train_model(model, train_data, test_data):

    train_dataloader = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)
    validation_dataloader = DataLoader(test_data, batch_size=model.batch_size, shuffle=True)

    optimizer_dict = {'SGD': torch.optim.SGD(model.parameters(), lr=model.lr)}
    optimizer = optimizer_dict[model.optimizer]

    train_loss =[]
    validation_loss = []
    for epoch in range(0, model.epochs):
        _train_loss = train_loop(model, train_dataloader, optimizer)
        _validation_loss, _accuracy = validation_loop(model, validation_dataloader)

        train_loss.append(_train_loss)
        validation_loss.append(validation_loss)

    