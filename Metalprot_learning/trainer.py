"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for model training and hyperparameter optimization.
"""

#imports
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

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
                path2output: str,
                training_data, 
                testing_data, 
                hyperparams: tuple):

    """Runs model training.

    Args:
        model (SingleLayerNet): SingleLayerNet object.
        path2output (str): Path to directory to contain output files.
        training_data (__main__.DistanceData): Dataset object of training data.
        testing_data (__main__.DistanceData): Dataset object of testing_data.
        hyperparams (tuple): Tuple containing hyperparams.
        seed (int, optional): The random seed for splitting. Defaults to 42.
    """

    #expand hyperparamters
    epochs, batch_size, lr, loss_fn, optimizer = hyperparams

    #instantiate dataloader objects for train and test sets
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    #define optimizer and loss function
    optimizer_dict = {'SGD': torch.optim.SGD(model.parameters(), lr=lr)}
    optimizer = optimizer_dict[optimizer]

    loss_fn_dict = {'MAE': torch.nn.L1Loss(),
                    'MSE': torch.nn.MSELoss(),
                    'SumOfSquaresLoss': SumOfSquaresLoss()}
    loss_fn = loss_fn_dict[loss_fn]

    train_loss =[]
    test_loss = []
    for epoch in range(0, epochs):
        print(f'Now on epoch {epoch}')
        _train_loss = train_loop(model, train_dataloader, loss_fn, optimizer)
        _test_loss = validation_loop(model, test_dataloader, loss_fn)

        train_loss.append(_train_loss)
        test_loss.append(_test_loss)

    torch.save(model.state_dict(), os.path.join(path2output, "model" + '.pth'))
    np.save(os.path.join(path2output, 'train_loss'), np.array(train_loss))
    np.save(os.path.join(path2output, 'test_loss'), np.array(test_loss))

def SumOfSquaresLoss(output, target):
    loss = torch.sum(torch.square(output - target))
    return loss