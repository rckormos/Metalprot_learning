#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs model training. 
"""

#imports
from Metalprot_learning.models import *

if __name__ == '__main__':
    path2observations = ''
    path2labels = ''

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
    training_proportion = 0.8
    seed = 42
    filename = None

    model = SingleLayerNet()
    train_model(model, path2observations, path2labels, epochs, batch_size, learning_rate, loss_function, 
                optimizer, filename=filename, train_prop=training_proportion, seed=seed)