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

def distribute_tasks(no_jobs: int, job_id: int, path2models: str):
    """Distributes batch jobs accross mutliple cores.

    Args:
        no_jobs (int): Number of jobs.
        job_id (int): Job id.
        path2models (str): Path to directory containing model subdirectories.

    Returns:
        tasks (list): List of models to be train by task.
    """

    models = [os.path.join(path2models, i) for i in os.listdir(path2models)]
    tasks = [models[i] for i in range(0,len(models)) if i % no_jobs == job_id]
    return tasks

def run_train(task: str, feature_file: str):

    try:
        with open(os.path.join(task, 'hyperparams.json'), 'r') as f:
            hyperparams = json.load(f)

    except:
        print(f'No hyperparams.json file found in {task}')
        return

    arch = hyperparams['arch']
    combination = (hyperparams['epochs'], hyperparams['batch_size'], hyperparams['lr'], hyperparams['loss_fn'], hyperparams['optimizer'])
    partition = hyperparams['partition']
    seed = hyperparams['seed']
    train_model(task, arch, feature_file, partition, seed, combination)

if __name__ == '__main__':
    path2output = sys.argv[1] #note that this is not actually where output files are written to. they are written to the model directories.
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    #provide paths to observations and labels
    path2features = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/datasetV1/compiled_features.pkl'
    path2models = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/models/MLP_v1'

    #distribute and run tasks
    tasks = distribute_tasks(no_jobs, job_id, path2models)
    for task in tasks:
        run_train(task, path2features) 