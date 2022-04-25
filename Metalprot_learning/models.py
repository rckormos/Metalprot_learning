"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains the architecture of the metalprot_learning model version 1. This model performs prediction of backbone distances to a metal.
"""

#imports
import numpy as np
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

class DistanceData(torch.utils.data.Dataset):
    "Custom dataset class"

    def __init__(self, observations, labels):
        self.labels = labels
        self.observations = observations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        observation = self.observations[index]
        label = self.labels[index]
        return observation, label 

def split_data(X, y, train_prop, seed):
    """Splits data into training and test sets.

    Args:
        X (np.ndarray): Observation data.
        y (np.ndarray): Label data.
        train_size (float): The proportion of data to be paritioned into the training set. 
        seed (int): The random seed for splitting.

    Returns:
        training_data (__main__.DistanceData): Dataset object of training data.
        validation_data (__main__.DistanceData): Dataset object of validation data.
    """

    training_size = int(train_prop * X.shape[0])
    indices = np.random.RandomState(seed=seed).permutation(X.shape[0])
    training_indices, val_indices = indices[:training_size], indices[training_size:]
    X_train, y_train, X_val, y_val = X[training_indices], y[training_indices], X[val_indices], y[val_indices]
    training_data, validation_data = DistanceData(X_train, y_train), DistanceData(X_val, y_val)

    return training_data, validation_data

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

def train_model(model, 
                observation_file: str, 
                label_file: str, 
                epochs: int, 
                batch_size: int, 
                lr: float, 
                loss_fn: str, 
                optimizer: str, 
                filename=None,
                 train_prop=0.8, 
                 seed=42):
    """Runs model training.

    Args:
        model (SingleLayerNet): SingleLayerNet object.
        observation_file (str): Path to observation matrix.
        label_file (str): Path to label matrix.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for training.
        loss_fn (str): Defines the loss function for backpropagation. Must be a string in {'MAE', 'MSE'}
        optimizer (str): Defines the optimization algorithm for backpropagation. Must be a string in {'SGD'}
        filename (str, optional): Filename to write trained model weights and biases to. Defaults to None.
        train_size (float, optional): The proportion of data to be paritioned into the training set. Defaults to 0.8.
        seed (int, optional): The random seed for splitting. Defaults to 42.
    """

    #split dataset into training and testing sets
    observations = np.load(observation_file)
    labels = np.load(label_file)
    train_data, test_data = split_data(observations, labels, train_prop, seed)

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

    