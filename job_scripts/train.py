#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs model training. 
"""

#imports
from Metalprot_learning.train.train import train_model
import os
import sys
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

if __name__ == '__main__':
    PATH2FEATURES = '/home/gpu/jzhang1198/data/ZN_binding_cores/cores-2022-07-10/compiled_features0.pkl'
    MODELS = [
    {
        'block_n1': {'pin': 40, 'pout': 8, 'kernel_size': 3, 'padding': 1, 'dropout': 0.2},
        'block0': {'pin': 12, 'pout': 64, 'kernel_size': 3, 'padding': 1, 'dropout': 0.3},
        'block1': {'pin_residual': 64, 'dilation_residual': 1, 'pin_conv': 64, 'pout_conv': 128, 'kernel_size_conv': 1, 'padding_conv': 0, 'kernel_size_pool': (2,2),'dropout': 0.2},
        'block2': {'pin_residual': 128, 'dilation_residual': 1, 'pin_conv': 128, 'pout_conv': 256, 'kernel_size_conv': 1, 'padding_conv': 0, 'kernel_size_pool': (2,2),'dropout': 0.2},
        'block3': {'pin_residual': 256, 'dilation_residual': 1, 'dropout': 0.2},
        'linear1': {'in': 2304, 'out': 512},
        'linear2': {'in': 512, 'out': 256},
        'linear3': {'in': 256, 'out': 48},
        'encodings': True,
        'batch_size': 16,
        'seed': 69,
        'lr': 0.0001,
        'epochs': 1
    }
    ]

    path2output, tasks = distribute_tasks(MODELS)
    for model in tasks:
        today = datetime.datetime.now()
        dirname = os.path.join(path2output, '-'.join(str(i) for i in [today.day, today.month, today.year, model['seed']]))
        os.mkdir(dirname)
        train_model(dirname, model, PATH2FEATURES)
        print(f'Execution time: {(datetime.datetime.now() - today)/ (60**2)} hours')