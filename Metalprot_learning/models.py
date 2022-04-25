"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains the architecture of the metalprot_learning model version 1. This model performs prediction of backbone distances to a metal.
"""

#imports
import torch
from torch import nn
from torch.utils.data import DataLoader

class SingleLayerNet(nn.Module):
    def __init__(self):
        super(SingleLayerNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(2544, 48),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.block1(x.float())
        return y

def train_loop(model, train_dataloader, loss_fn, optimizer):

    for batch, (X, y) in enumerate(train_dataloader):
        
        #make prediction
        prediction = model.forward(X) 
        loss = loss_fn(prediction, y)

        #backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def validation_loop(model, validation_dataloader, loss_fn):
    validation_loss = 0
    with torch.no_grad():
        for X,y in validation_dataloader:
            prediction = model(X)
            validation_loss += loss_fn(prediction,y).item()

    validation_loss /= len(validation_dataloader)
    return validation_loss

def train_model(model, train_data, test_data, epochs: int, batch_size: int, lr: float, loss_fn: str, optimizer: str, filename=None):
    """Runs model training.

    Args:
        model (SingleLayerNet): SingleLayerNet object.
        train_data: DataLoader object containing training data.
        test_data: Dataloader object containing validation data.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for training.
        loss_fn (str): Defines the loss function for backpropagation. Must be a string in {'MAE', 'MSE'}
        optimizer (str): Defines the optimization algorithm for backpropagation. Must be a string in {'SGD'}
    """

    #instantiate dataloader objects for train and test sets
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    #define optimizer and loss function
    optimizer_dict = {'SGD': torch.optim.SGD(model.parameters(), lr=lr)}
    optimizer = optimizer_dict[optimizer]

    loss_fn_dict = {'MAE': torch.nn.L1Loss(),
                    'MSE': torch.nn.MSELoss()}
    loss_fn = loss_fn_dict[loss_fn]

    train_loss =[]
    validation_loss = []
    for epoch in range(0, epochs):
        _train_loss = train_loop(model, train_dataloader, loss_fn, optimizer)
        _validation_loss = validation_loop(model, validation_dataloader, loss_fn)

        train_loss.append(_train_loss)
        validation_loss.append(_validation_loss)

    if filename:
        torch.save(model.state_dict(), filename + '.pth')

    return train_loss, validation_loss

    