#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs model hyperparameter tuning.
"""

#imports
import os
import sys
import json
import torch
import optuna
import datetime
import numpy as np
from Metalprot_learning.train import models, train

def distribute_tasks():
    path2output = sys.argv[1]

    today = datetime.datetime.now()
    path2output = os.path.join(path2output, '_'.join( str(i) for i in ['tune', today.day, today.month, today.year, today.hour, today.minute, today.second, today.microsecond]))
    os.mkdir(path2output)
    
    return path2output

def write_output_files(subdir: str, params: tuple, model, train_loss: np.ndarray, test_loss: np.ndarray):
    np.save(os.path.join(subdir, 'train_loss.npy'), train_loss)
    np.save(os.path.join(subdir, 'test_loss.npy'), test_loss)

    config = {'seed': params[0],
        'batch_size': params[1],
        'lr': params[2],
        'l1': params[3],
        'l2': params[4],
        'l3': params[5],
        'input_dropout': params[6],
        'hidden_dropout': params[7]}

    with open(os.path.join(subdir, 'config.json'), 'w') as f:
        json.dump(config, f)

    torch.save(model.state_dict, os.path.join(subdir, 'model.pth'))

if __name__ == '__main__':

    FEATURES_FILE = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/datasetV3/compiled_features.pkl'
    INPUT_DIM = 2544
    OUTPUT_DIM = 48

    path2output = distribute_tasks()

    #define objective function
    def objective(trial):
        trial_dir = os.path.join(path2output, str(trial.number))
        os.mkdir(trial_dir)

        seed = np.random.randint(0,1000)
        batch_size = 51
        lr = 0.0346838274787568
        l1 = trial.suggest_int("l1", 2400, 2500)
        l2 = trial.suggest_int("l2", 1100, 1200)
        l3 = trial.suggest_int("l3", 600,700)
        input_dropout = trial.suggest_float("input_dropout",0.1, 0.99)
        hidden_dropout = trial.suggest_float("hidden_dropout", 0.1, 0.99)

        model = models.DoubleLayerNet(INPUT_DIM, l1, l2, l3, OUTPUT_DIM, input_dropout, hidden_dropout)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        print(f'Training running on {device}')

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = torch.nn.L1Loss()

        train_dataloader, test_dataloader, validation_dataloader = train.load_data(FEATURES_FILE, (0.8,0.1,0.1), batch_size, seed, False)

        train_loss = np.array([])
        test_loss = np.array([])
        for epoch in range(0, 40):
            _train_loss = train.train_loop(model, train_dataloader, loss_fn, optimizer, device)
            _test_loss = train.validation_loop(model, test_dataloader, loss_fn, device)

            train_loss = np.append(train_loss, _train_loss)
            test_loss = np.append(test_loss, _test_loss)
            trial.report(_test_loss, epoch)

            if trial.should_prune():
                write_output_files(trial_dir, (seed, batch_size, lr, l1, l2, l3, input_dropout, hidden_dropout), model, train_loss, test_loss)
                raise optuna.exceptions.TrialPruned()

        write_output_files(trial_dir, (seed, batch_size, lr, l1, l2, l3, input_dropout, hidden_dropout), model, train_loss, test_loss)
        return _test_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    importances = optuna.importance.get_param_importances(study)

    with open(os.path.join(path2output, 'importances.json'), 'w') as f:
        json.dump(importances, f)


