#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs model training. 
"""

#imports
from Metalprot_learning.models import *

if __name__ == '__main__':
    path2observations = '/Users/jonathanzhang/Documents/ucsf/degrado/data/observations.npy'
    path2labels = '/Users/jonathanzhang/Documents/ucsf/degrado/data/labels.npy'

    #define architecture of neural network
    arch = [{'input_dim': 2544, 'output_dim': 1272, 'activation': None}, 
            {'input_dim': 1272, 'output_dim': 636, 'activation': None},
            {'input_dim': 636, 'output_dim': 318, 'activation': None},
            {'input_dim': 318, 'output_dim': 48, 'activation': 'ReLU'}]


    #define hyperparameters
    epochs = 1000
    batch_size = 4
    learning_rate = 0.001
    loss_function = 'MAE' #can be mean absolute error (MAE) or mean squared error (MSE)
    optimizer = 'SGD' #currently can only be stochastic gradient descent (SGD)
    partition = (0.8,0.1,0.1)
    seed = 42
    filename = '/Users/jonathanzhang/Documents/ucsf/degrado/data/deezma.pth'

    model = SingleLayerNet(arch)
    train_model(model, path2observations, path2labels, epochs, batch_size, learning_rate, loss_function, 
                optimizer, filename=filename, partition=partition, seed=seed)