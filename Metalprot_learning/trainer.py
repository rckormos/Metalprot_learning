"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for model training and hyperparameter optimization.
"""

#imports
import numpy as np
import torch

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
                observation_file: str, 
                label_file: str, 
                epochs: int, 
                batch_size: int, 
                lr: float, 
                loss_fn: str, 
                optimizer: str, 
                name=None,
                partition=(0.8,0.1,0.1), 
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
        name (str, optional): Filename to write trained model weights and biases to. Defaults to None.
        train_size (float, optional): The proportion of data to be paritioned into the training set. Defaults to 0.8.
        seed (int, optional): The random seed for splitting. Defaults to 42.
    """

    #split dataset into training and testing sets
    observations = np.load(observation_file)
    labels = np.load(label_file)
    train_data, test_data, X_val, y_val = split_data(observations, labels, partition, seed)

    #instantiate dataloader objects for train and test sets
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

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


    torch.save(model.state_dict(), os.path.join(path2output, name + '.pth'))
    np.save(os.path.join(path2output, name + '_train_loss'), np.array(train_loss))
    np.save(os.path.join(path2output, name + '_test_loss'), np.array(test_loss))
    np.save(os.path.join(path2output, 'X_val'), X_val)
    np.save(os.path.join(path2output, 'y_val'), y_val)