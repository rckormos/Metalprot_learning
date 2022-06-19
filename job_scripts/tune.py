#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs model hyperparameter tuning.
"""

#imports
import os
import sys
import json
import optuna
import datetime
import numpy as np
from Metalprot_learning.tune import define_objective

if __name__ == '__main__':
    path2output = sys.argv[1]

    #user-defined variables
    FEATURES_FILE = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/datasetV3/compiled_features.pkl'
    CONFIG = {'seed': np.randint(0,1000),
        'batch_size': 51,
        'lr': 0.0346838274787568,
        'l1': (2400, 2500),
        'l2': (1100, 1200),
        }
    ENCODINGS = False

    #create output directory to hold data from experiment
    today = datetime.datetime.now()
    dirname = os.path.join(path2output, '_'.join( str(i) for i in ['tune', today.day, today.month, today.year, today.hour, today.minute, today.second, today.microsecond]))
    os.mkdir(dirname)

    objective = define_objective(dirname, FEATURES_FILE, CONFIG, ENCODINGS)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    importances = optuna.importance.get_param_importances(study)

    with open(os.path.join(dirname, 'importances.json'), 'w') as f:
        json.dump(importances, f)