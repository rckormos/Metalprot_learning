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

def configure_run(models: dict, cbeta: bool, encodings: bool, noise: bool):

    encoding_dim = 20*12 if encodings else 0
    distance_dim = 3600 if cbeta else 2304
    total_dim = encoding_dim + distance_dim
    output_dim = 60 if cbeta else 48
    path2features = '' if cbeta else ''

    for model in models:
        model['input'] = total_dim
        model['output'] = output_dim
        model['encodings'] = encodings
        model['noise'] = noise

    return models, path2features

if __name__ == '__main__':

    CBETA = False
    ENCODINGS = True
    NOISE = False
    MODELS = [
        {'l1': 2789,
        'l2': 1725,
        'l3': 777,
        'input_dropout': 0.15578875454945562,
        'hidden_dropout': 0.30066048849068494,
        'weight_decay': 0,
        'batch_size': 51,
        'lr': 0.003853602505520586,
        'seed': np.random.randint(1000),
        'epochs': 2000,
        'loss_fn': 'MAE'}
    ]

    path2output, tasks = distribute_tasks(MODELS)
    tasks, path2features = configure_run(tasks, CBETA, ENCODINGS, NOISE)
    for model in tasks:
        today = datetime.datetime.now()
        dirname = os.path.join(path2output, '_'.join(str(i) for i in [today.day, today.month, today.year, today.hour, today.minute, today.second, today.microsecond, model['seed']]))
        os.mkdir(dirname)
        train_model(dirname, model, path2features)