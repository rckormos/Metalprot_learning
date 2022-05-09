#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs model training. 
"""

#imports
from Metalprot_learning.trainer.model_trainer import train_model
import sys
import os
import json

def distribute_tasks(no_jobs: int, job_id: int, epochs: list, batch_sizes: list, learning_rates: list, 
                    loss_functions: list, optimizers: list):
    """Distributes batch jobs accross mutliple cores.

    Args:
        no_jobs (int): Number of jobs.
        job_id (int): Job id.
        epochs (list): List of epochs.
        batch_size (list): List of batch sizes.
        learning_rate (list): List of learning rates.
        loss_function (list): List of loss functions to be tested.
        optimizer (list): List of optimizers to be tested.

    Returns:
        tasks (list): List of tuples containing hyperparameter combinations.
    """

    combinations = list(set([(i,j,k,l,m) for i in epochs for j in batch_sizes for k in learning_rates for l in loss_functions for m in optimizers]))
    combinations.sort()
    tasks = [combinations[i] for i in range(0,len(combinations)) if i % no_jobs == job_id]
    
    return tasks

def run_train(path2output: str, arch: dict, observation_file: str, label_file: str, index_file: str, partition: tuple, seed: int, combination: tuple):
    name = '_'.join([str(i) for i in list(combination)])

    model_dir = os.path.join(path2output, name)
    os.makedirs(model_dir)

    train_model(path2output, arch, observation_file, label_file, index_file, partition, seed, combination)

    with open(os.path.join(model_dir, 'architecture.json'), 'w') as f:
        json.dump(arch, f)

if __name__ == '__main__':
    path2output = sys.argv[1] 
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    #provide paths to observations and labels
    path2observations = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/datasetV1/observations.npy'
    path2labels = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/datasetV1/labels.npy'
    path2index = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/datasetV1/index.pkl'

    #define architecture of neural network
    arch = [{'input_dim': 2544, 'output_dim': 1272, 'activation': 'ReLU'}, 
            {'input_dim': 1272, 'output_dim': 636, 'activation': 'ReLU'},
            {'input_dim': 636, 'output_dim': 318, 'activation': 'ReLU'},
            {'input_dim': 318, 'output_dim': 48, 'activation': 'ReLU'}]

    #define hyperparameters. if you would like to implement a grid search, simply add more values to the lists
    epochs = [2000]
    batch_sizes = [1000]
    learning_rates = [0.01]
    loss_functions = ['MAE'] #can be mean absolute error (MAE) or mean squared error (MSE)
    optimizers = ['SGD'] #currently can only be stochastic gradient descent (SGD)
    partition = (0.8,0.1,0.1)
    seed = 42

    #distribute and run tasks
    tasks = distribute_tasks(no_jobs, job_id, epochs, batch_sizes, learning_rates, loss_functions, optimizers)
    for combination in tasks:
        run_train(path2output, combination)