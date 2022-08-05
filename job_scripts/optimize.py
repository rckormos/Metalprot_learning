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

def distribute_tasks(no_jobs: int, job_id: int, arch: list, input_layer_rates: np.ndarray, hidden_layer_rates: np.ndarray):
    combinations = [(i,j) for i in input_layer_rates for j in hidden_layer_rates]
    architectures = [arch] * len(combinations)
    updated = []

    for i, comb in enumerate(combinations):
        _arch = deepcopy(architectures[i]) #need to copy due to the way python stores dictionaries and lists in memory
        input_rate, hidden_rate = comb

        for j, layer in enumerate(_arch):
            if j == 0:
                layer['dropout'] = input_rate
            elif j == len(_arch)-1:
                layer['dropout'] = 0
            else:
                layer['dropout'] = hidden_rate

        updated.append(_arch)

    tasks = [updated[i] for i in range(0,len(updated)) if i % no_jobs == job_id]
    return tasks

if __name__ == '__main__':
    path2output = sys.argv[1] 
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    FEATURES_FILE = '/home/gpu/jzhang1198/data/ZN_binding_cores/cnn-cores-2022-07-31/compiled_data.pkl'
    NO_TRIALS = 30
    CONFIG = {
        'block_n1': {'in': 40, 'out': (2,20), 'kernel_size': (1,10), 'padding': (1,5), 'dropout': (0,.5)},
        'block0': {'out': 64, 'kernel_size': 3, 'padding': 1, 'dropout': 0.3},
        'block1': {'dilation_residual': 1, 'out': 128, 'kernel_size_conv': 1, 'padding_conv': 0, 'kernel_size_pool': (2,2),'dropout': 0.2},
        'block2': {'dilation_residual': 1, 'out': 256, 'kernel_size_conv': 1, 'padding_conv': 0, 'kernel_size_pool': (2,2),'dropout': 0.2},
        'block3': {'dilation_residual': 1, 'dropout': 0.2},
        'linear1': {'out': 512},
        'linear2': {'out': 256},
        'linear3': {'out': 48},
        'encodings': True,
        'batch_size': 16,
        'seed': 69,
        'lr': 0.0001,
        'epochs': 1
        }

    today = datetime.datetime.now()
    dirname = os.path.join(path2output, '-'.join( str(i) for i in ['tune', today.year, today.month, today.day, today.hour, today.minute, today.second, today.microsecond]))
    os.mkdir(dirname)
    objective = tune.define_objective(path2output, FEATURES_FILE, CONFIG)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=NO_TRIALS)

    if NO_TRIALS > 1:
        importances = optuna.importance.get_param_importances(study)
        with open(os.path.join(dirname, 'importances.json'), 'w') as f:
            json.dump(importances, f)