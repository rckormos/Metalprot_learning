#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script creates directories and architecture files that specify model hyperparameters.
"""

#impots 
import sys
import os
import json

def enumerate_models(arch: dict, epochs: list, batch_sizes: list, learning_rates: list, loss_functions: list, optimizers: list, partition: tuple, seed: int):
    """Distributes batch jobs accross mutliple cores.

    Args:
        arch (dict): Dictionary defining model architecture.
        epochs (list): List of epochs.
        batch_size (list): List of batch sizes.
        learning_rate (list): List of learning rates.
        loss_function (list): List of loss functions to be tested.
        optimizer (list): List of optimizers to be tested.
        partition (tuple): Defines partitioning of data into training, testing, and validation sets.
        seed (int): Random seed for reproducibility.

    Returns:
        combinations (list): List of tuples containing hyperparameter combinations.
    """

    combinations = list(set([(i,j,k,l,m) for i in epochs for j in batch_sizes for k in learning_rates for l in loss_functions for m in optimizers]))
    combinations.sort()    

    for combination in combinations:
        #make directory to hold training hyperparameters
        name = '_'.join([str(i) for i in list(combination)])
        model_dir = os.path.join(path2output, name)
        os.makedirs(model_dir)

        hyperparams = {}
        hyperparams['arch'] = arch
        hyperparams['epochs'] = combination[0]
        hyperparams['batch_size'] = combination[1]
        hyperparams['lr'] = combination[2]
        hyperparams['loss_fn'] = combination[3]
        hyperparams['optimizer'] = combination[4]
        hyperparams['partition'] = partition
        hyperparams['seed'] = seed

        with open(os.path.join(model_dir, 'hyperparams.json'), 'w') as f:
            json.dump(hyperparams, f)

if __name__ == '__main__':
    path2output = sys.argv[1]

    #define architecture of neural network
    arch = [{'input_dim': 2544, 'output_dim': 1000, 'activation': 'ReLU', 'dropout': None}, 
            {'input_dim': 1000, 'output_dim': 500, 'activation': 'ReLU', 'dropout': None},
            {'input_dim': 500, 'output_dim': 48, 'activation': 'ReLU', 'dropout': None}]

    #define hyperparameters. if you would like to implement a grid search, simply add more values to the lists
    epochs = [2003]
    batch_sizes = [1000]
    learning_rates = [0.01]
    loss_functions = ['MAE'] #can be mean absolute error (MAE) or mean squared error (MSE)
    optimizers = ['SGD'] #currently can only be stochastic gradient descent (SGD)
    partition = (0.8,0.1,0.1)
    seed = 42

    enumerate_models(arch, epochs, batch_sizes, learning_rates, loss_functions, optimizers, partition, seed)
