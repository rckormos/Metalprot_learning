#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script optimizes model performance by tuning dropout rate. 
"""

#imports
import os
import sys
import json
import optuna
import datetime
import numpy as np
from Metalprot_learning.train import tune

def distribute_tasks(no_jobs: int, job_id: int, no_studies: int):
    tasks = [x for x in range(no_studies) if x % no_jobs == job_id]
    return len(tasks)

if __name__ == '__main__':
    path2output = sys.argv[1]
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    FEATURES_FILE = '/Users/jonathanzhang/Documents/ucsf/degrado/data/metalprot_learning/cnn/ZN_binding_cores/cnn-cores-2022-07-31/compiled_data.pkl'
    NO_STUDIES = 2
    NO_TRIALS = 1
    CONFIG = {
        'block_n1': {'in': 40, 'out': (2,20), 'padding': (2,5), 'dropout': (0,.5)},
        'block0': {'out': (50,100), 'kernel_size': (1,4), 'padding': (0,5), 'dropout': (0.1, 0.9)}, #constraint: kernel size must be less than the width of the block_n1 output
        'block1': {'dilation_residual': (1,2), 'out': (100,200), 'kernel_size': (1,2), 'padding': (1,2), 'kernel_size_pool': 2,'dropout': (0.1, 0.9)},
        'block2': {'dilation_residual': (1,2), 'out': (200,400), 'kernel_size': (1,2), 'padding': (1,2), 'kernel_size_pool': 2,'dropout': (0.1, 0.9)},
        'block3': {'dilation_residual': (1,2), 'dropout': (0.1, 0.9)},
        'linear1': {'out': (200,400)},
        'linear2': {'out': (200,400)},
        'linear3': {'out': 48},
        'encodings': True,
        'batch_size': (2,40),
        'seed': 69,
        'lr': (0.0001, 0.01),
        'epochs': 1
        }

    today = datetime.datetime.now()
    dirname = os.path.join(path2output, '-'.join( str(i) for i in ['tune', today.year, today.month, today.day, today.hour, today.minute, today.second, today.microsecond]))
    os.mkdir(dirname)
    for study in range(distribute_tasks(no_jobs, job_id, NO_STUDIES)):
        study_dirname = os.path.join(dirname, 'study' + '_' + str(study))
        os.mkdir(study_dirname)
        objective = tune.define_objective(study_dirname, FEATURES_FILE, CONFIG)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=NO_TRIALS)

        if NO_TRIALS > 1:
            importances = optuna.importance.get_param_importances(study)
            with open(os.path.join(study_dirname, 'importances.json'), 'w') as f:
                json.dump(importances, f)