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

    FEATURES_FILE = '/home/gpu/jzhang1198/data/ZN_binding_cores/cnn-cores-2022-07-31/compiled_data.pkl'
    NO_STUDIES = 3
    NO_TRIALS = 20
    CONFIG = {
        'block_n1': {'in': 40, 'out': 8, 'padding': 1, 'dropout_n1': (0.1, 0.6)},
        'block0': {'out': 64, 'kernel_size': 3, 'padding': 1, 'dropout_0': (0.3, 0.8)}, 
        'block1': {'dilation_residual': 1, 'out': 128, 'kernel_size': 1, 'padding': 0, 'kernel_size_pool': 2,'dropout_1': (0.3, 0.8)},
        'block2': {'dilation_residual': 1, 'out': 256, 'kernel_size': 1, 'padding': 0, 'kernel_size_pool': 2,'dropout_2': (0.3, 0.8)},
        'block3': {'dilation_residual': 1, 'dropout_2': (0.3, 0.8)},
        'linear1': {'out': 512},
        'linear2': {'out': 256},
        'linear3': {'out': 48},
        'encodings': True,
        'batch_size': 14,
        'seed': 69,
        'lr': 0.00020894227630869483,
        'epochs': 30
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