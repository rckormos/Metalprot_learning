"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for model training and hyperparameter optimization.
"""

#imports
import os
import numpy as np
import json
import torch
from Metalprot_learning.trainer import datasets
from Metalprot_learning.trainer import models

def load_data(features_file: str, partitions: tuple, batch_size: int, seed: int):
    """Loads data for model training.

    Args:
        feature_file (str): Path to compiled_features.pkl file.
        partitions (tuple): Tuple containing percentages of the dataset partitioned into training, testing, and validation sets respectively.
        batch_size (int): The batch size.
        seed (int): Random seed defined by user.

    Returns:
        train_dataloader (torch.utils.data.DataLoader): DataLoader object containing shuffled training observations and labels.
        test_dataloader (torch.utils.data.DataLoader): DataLoader object containing shuffled testing observations and labels.
    """
    training_data, testing_data, _ = datasets.split_data(features_file, partitions, seed)

    training_observations, training_labels, _ = training_data
    train_dataloader = torch.utils.data.DataLoader(datasets.DistanceData(training_observations, training_labels), batch_size=batch_size, shuffle=True)
    
    testing_observations, testing_labels, _ = testing_data
    test_dataloader = torch.utils.data.DataLoader(datasets.DistanceData(testing_observations, testing_labels), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def train_loop(model, train_dataloader, loss_fn, optimizer):
    """Runs a single epoch of model training.

    Args:
        model: Instantiation of a neural network to be trained.
        train_dataloader (torch.utils.data.DataLoader): Dataloader containing training data.
        loss_fn: User-defined loss function.
        optimizer: User-defined optimizer for backpropagation.

    Returns:
        train_loss: The average training loss across batches.
    """

    for batch, (X, y) in enumerate(train_dataloader):
        
        #make prediction
        prediction = model.forward(X) 
        loss = loss_fn(prediction, y)

        #backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = loss.item()
    return train_loss

def validation_loop(model, test_dataloader, loss_fn):
    """Computes a forward pass of the testing dataset through the network and the resulting testing loss.

    Args:
        model: Instantiation of a neural network to be trained.
        test_dataloader (torch.utils.data.DataLoader): Dataloader containing testing data.
        loss_fn: User-defined loss function.

    Returns:
        validation_loss: The average validation loss across batches.
    """

    validation_loss = 0
    with torch.no_grad():
        for X,y in test_dataloader:
            prediction = model(X)
            validation_loss += loss_fn(prediction,y).item()

    validation_loss /= len(test_dataloader)
    return validation_loss

def train_model(path2output: str, arch: dict, features_file: str, partitions: tuple, seed: int, hyperparams: tuple):
    """Runs model training.

    Args:
        path2output (str): Directory to dump output files.
        arch (dict): List of dictionaries defining architecture of the neural network.
        observation_file (str): Path to observation matrix file.
        label_file (str): Path to label matrix file.
        partitions (tuple): Tuple containing percentages of the dataset partitioned into training, testing, and validation sets respectively.
        index_file (str): Path to index file.
        seed (int): Random seed defined by user.
        hyperparams (tuple): Tuple containing hyperparameters for model training.
    """

    #expand hyperparamters and instantiate model
    epochs, batch_size, lr, loss_fn, optimizer = hyperparams
    model = models.SingleLayerNet(arch)

    #instantiate dataloader objects for train and test sets
    train_dataloader, test_dataloader = load_data(features_file, partitions, batch_size, seed)

    #define optimizer and loss function
    optimizer_dict = {'SGD': torch.optim.SGD(model.parameters(), lr=lr)}
    optimizer = optimizer_dict[optimizer]
    loss_fn_dict = {'MAE': torch.nn.L1Loss(),
                    'MSE': torch.nn.MSELoss()}
    loss_fn = loss_fn_dict[loss_fn]

    train_loss =[]
    test_loss = []
    for epoch in range(0, epochs):
        print(f'Now on epoch {epoch}')
        _train_loss = train_loop(model, train_dataloader, loss_fn, optimizer)
        _test_loss = validation_loop(model, test_dataloader, loss_fn)

        train_loss.append(_train_loss)
        test_loss.append(_test_loss)

    #write output files
    torch.save(model.state_dict(), os.path.join(path2output, "model" + '.pth'))
    np.save(os.path.join(path2output, 'train_loss'), np.array(train_loss))
    np.save(os.path.join(path2output, 'test_loss'), np.array(test_loss))
    with open(os.path.join(path2output, 'architecture.json'), 'w') as f:
        json.dump(arch, f)