"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for model training and hyperparameter optimization.
"""

#imports
import numpy as np
import torch
from ray import tune
from functools import partial
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

def train_loop(model, train_dataloader, loss_fn, optimizer, device):
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
        X, y = X.to(device), y.to(device)
        
        #make prediction
        prediction = model.forward(X) 
        loss = loss_fn(prediction, y)

        #backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = loss.item()
    return train_loss

def validation_loop(model, test_dataloader, loss_fn, device):
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
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            validation_loss += loss_fn(prediction,y).item()

    validation_loss /= len(test_dataloader)
    return validation_loss

def train_full_model(config: dict, input=int, output=int, features_file=str, seed=int):
    """Runs model training.

    Args:
        path2output (str): Directory to dump output files.
        arch (dict): List of dictionaries defining architecture of the neural network.
        features_file (str): Contains observations and labels.
        config (dict): Defines configurable model hyperparameters.
        arch (dict): Defines the architecture of the neural network with configurable parameters.
        partitions (tuple): Tuple containing percentages of the dataset partitioned into training, testing, and validation sets respectively.
        seed (int): Random seed defined by user.
    """

    torch.manual_seed(seed)

    #instantiate model
    model = models.SingleLayerNetV2(input, config['l1'], config['l2'], output) if 'l3' not in config.keys() else models.DoubleLayerNet(input, config['l1'], config['l2'], config['l3'], output)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    #instantiate dataloader objects for train and test sets
    train_dataloader, test_dataloader = load_data(features_file, (0.8,0.1,0.1), config['batch_size'], seed)

    #define optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    loss_fn = torch.nn.L1Loss()

    for epoch in range(0, config['epochs']):
        _train_loss = train_loop(model, train_dataloader, loss_fn, optimizer, device)
        _test_loss = validation_loop(model, test_dataloader, loss_fn, device)

        tune.report(test_loss=_test_loss, train_loss=_train_loss)

def tune_model(path2output: str, seed: int, no_samples: int, config: dict, coordinating_resis: int, no_neighbors: int, features_file: str, cpus=1, gpus=0, max_epochs=3000):
    no_resis = coordinating_resis + (2*no_neighbors*coordinating_resis)
    input_dim = (4*no_resis)**2 + (20*no_resis)
    output_dim = 4*no_resis

    scheduler = tune.schedulers.ASHAScheduler(metric='test_loss', mode='min', max_t=max_epochs, grace_period=1, reduction_factor=2)
    result = tune.run(
        partial(train_full_model, input=input_dim, output=output_dim, features_file=features_file, seed=seed),
        resources_per_trial={'cpu': cpus, "gpu": gpus},
        config=config,
        num_samples=no_samples,
        scheduler=scheduler,
        local_dir=path2output)

