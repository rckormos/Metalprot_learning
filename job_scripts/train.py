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

if __name__ == '__main__':
    PATH2FEATURES = '/home/gpu/jzhang1198/data/ZN_binding_cores/cores-2022-07-10/compiled_features_no2coords.pkl'
    MODELS = [
        {'l1': 1947,
        'l2': 652,
        'l3': 562,
        'l4': 300,
        'input_dropout': 0.3,
        'hidden_dropout': 0.4,
        'weight_decay': 0,
        'batch_size': 20,
        'lr': 0.0092982,
        'seed': np.random.randint(1000),
        'epochs': 2000,
        'loss_fn': 'MAE',
        'c_beta': True,
        'encodings': True,
        'noise': False},
    ]

    path2output, tasks = distribute_tasks(MODELS)
    for model in tasks:
        today = datetime.datetime.now()
        dirname = os.path.join(path2output, '-'.join(str(i) for i in [today.day, today.month, today.year, model['seed']]))
        os.mkdir(dirname)
        train_model(dirname, model, PATH2FEATURES)
        print(f'Execution time: {(datetime.datetime.now() - today)/ (60**2)} hours')