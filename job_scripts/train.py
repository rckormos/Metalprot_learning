#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs model training. 
"""

#imports
from Metalprot_learning.train.train import train_model
import os
import sys
import numpy as np
import datetime

def distribute_tasks(MODELS: list):
    """Distributes batch jobs accross mutliple cores.

    Args:
        no_jobs (int): Number of jobs.
        job_id (int): Job id.
        path2models (str): Path to directory containing model subdirectories.

    Returns:
        tasks (list): List of models to be train by task.
    """

    path2output = sys.argv[1] #note that this is not actually where output files are written to. they are written to the model directories.
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    tasks = [MODELS[i] for i in range(0,len(MODELS)) if i % no_jobs == job_id]
    return path2output, tasks

<<<<<<< HEAD
    #define hyperparameters. if you would like to implement a grid search, simply add more values to the lists
    epochs = [1]
    batch_sizes = [1000]
    learning_rates = [0.01]
    loss_functions = ['MAE'] #can be mean absolute error (MAE) or mean squared error (MSE)
    optimizers = ['SGD'] #currently can only be stochastic gradient descent (SGD)
    partition = (0.8,0.1,0.1)
    seed = 42

    #load data
    training_data, testing_data, validation_data = split_data(path2observations, path2labels, partition, seed)

    #distribute and run tasks
    tasks = distribute_tasks(no_jobs, job_id, epochs, batch_sizes, learning_rates, loss_functions, optimizers)
    for combination in tasks:
        run_train(path2output, combination)
=======
if __name__ == '__main__':
    PATH2FEATURES = '/home/gpu/jzhang1198/data/ZN_binding_cores/cores-2022-07-10/compiled_features0.pkl'
    MODELS = [
        {'l1': 2000,
        'l2': 1700,
        'l3': 400,
        'input_dropout': 0.15578875454945562,
        'hidden_dropout': 0.30066048849068494,
        'weight_decay': 0,
        'batch_size': 50,
        'lr': 0.003,
        'seed': np.random.randint(1000),
        'epochs': 2000,
        'loss_fn': 'MAE',
        'c_beta': True,
        'encodings': True,
        'noise': True},

        {'l1': 2000,
        'l2': 1700,
        'l3': 400,
        'input_dropout': 0.15578875454945562,
        'hidden_dropout': 0.30066048849068494,
        'weight_decay': 0,
        'batch_size': 50,
        'lr': 0.003,
        'seed': np.random.randint(1000),
        'epochs': 2000,
        'loss_fn': 'MAE',
        'c_beta': True,
        'encodings': True,
        'noise': False}

    ]

    path2output, tasks = distribute_tasks(MODELS)
    for model in tasks:
        today = datetime.datetime.now()
        dirname = os.path.join(path2output, '-'.join(str(i) for i in [today.day, today.month, today.year, model['seed']]))
        os.mkdir(dirname)
        train_model(dirname, model, PATH2FEATURES)
        print(f'Execution time: {(datetime.datetime.now() - today)/ (60**2)} hours')
>>>>>>> d69c730f6ccba4dbadb6bec84bbc4d8f4ce728c7
