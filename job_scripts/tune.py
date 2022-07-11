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
    features_file = '/home/gpu/jzhang1198/data/ZN_binding_cores/cores-2022-07-10/compiled_features0.pkl'
    config = {'seed': np.random.randint(0,1000),
        'batch_size': (20, 100),
        'lr': (0.0001, 0.01),
        'l1': (1000, 2000),
        'l2': (500, 1000),
        'l3': (100, 600),
        'input_dropout': (0.1, 0.6),
        'hidden_dropout': (0.1, 0.8),
        'epochs': 60,
        'weight_decay': 0,
        'optimizer_key': 0,
        'loss_fn_key': 0,
        'encodings': True,
        'noise': True,
        'c_beta': True
        }

    #create output directory to hold data from experiment
    today = datetime.datetime.now()
    dirname = os.path.join(path2output, '_'.join( str(i) for i in ['tune', today.day, today.month, today.year, today.hour, today.minute, today.second, today.microsecond]))
    os.mkdir(dirname)

    objective = define_objective(dirname, features_file, config)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    importances = optuna.importance.get_param_importances(study)

    with open(os.path.join(dirname, 'importances.json'), 'w') as f:
        json.dump(importances, f)