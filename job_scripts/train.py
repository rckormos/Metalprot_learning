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

def distribute_tasks(no_jobs: int, job_id: int, MODELS: list):
    """Distributes batch jobs accross mutliple cores.

    Args:
        no_jobs (int): Number of jobs.
        job_id (int): Job id.
        path2models (str): Path to directory containing model subdirectories.

    Returns:
        tasks (list): List of models to be train by task.
    """
    tasks = [MODELS[i] for i in range(0,len(MODELS)) if i % no_jobs == job_id]
    return tasks

if __name__ == '__main__':
    path2output = sys.argv[1] #note that this is not actually where output files are written to. they are written to the model directories.
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    #provide paths to observations and labels
    PATH2FEATURES = '/home/gpu/jzhang1198/data/ZN_binding_cores/datasetV2/compiled_features.pkl'
    MODELS = [
        {'input': 2544,
        'l1': 2458,
        'l2': 1168,
        'l3': 621,
        'output': 48,
        'batch_size': 51,
        'lr': 0.0346838274787568,
        'seed': np.random.randint(0,1000),
        'epochs': 2000}
    ]

    for model in MODELS:
        today = datetime.datetime.now()
        dirname = os.path.join(path2output, '_'.join(str(i) for i in [today.day, today.month, today.year, today.hour, today.minute, today.second, today.microsecond, model['seed']]))
        os.mkdir(dirname)
        train_model(dirname, model, PATH2FEATURES)




    
    
