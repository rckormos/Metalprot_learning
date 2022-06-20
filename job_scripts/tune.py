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
from Metalprot_learning.train import models
from Metalprot_learning.train import train
from Metalprot_learning.tune import define_objective

def _write_output_files(subdir: str, train_loss: np.ndarray, test_loss: np.ndarray):
    np.save(os.path.join(subdir, 'train_loss.npy'), train_loss)
    np.save(os.path.join(subdir, 'test_loss.npy'), test_loss)

if __name__ == '__main__':
    path2output = sys.argv[1]

    #user-defined variables
    features_file = '/home/gpu/jzhang1198/data/ZN_binding_cores/datasetV4/barcoded_compiled_features.pkl'
    config = {'seed': np.random.randint(0,1000),
        'batch_size': 51,
        'lr': 0.0346838274787568,
        'l1': 2458,
        'l2': 1168,
        'l3': 621,
        'input_dropout': (0.1, 0.6),
        'hidden_dropout': (0.1, 0.8),
        'epochs': 60,
        'weight_decay': 0,
        'optimizer_key': 0,
        'loss_fn_key': 0
        }
    encodings = True

    #create output directory to hold data from experiment
    today = datetime.datetime.now()
    dirname = os.path.join(path2output, '_'.join( str(i) for i in ['tune', today.day, today.month, today.year, today.hour, today.minute, today.second, today.microsecond]))
    os.mkdir(dirname)

    # objective = define_objective(dirname, FEATURES_FILE, CONFIG, ENCODINGS)

    input_dim = 2544 if encodings else 2304
    output_dim = 48

    def objective(trial):

        trial_dir = os.path.join(dirname, str(trial.number))
        os.mkdir(trial_dir)

        seed = np.random.randint(0,1000)
        np.random.seed(seed)
        torch.manual_seed(seed)

        batch_size = trial.suggest_int("batch_size", config['batch_size'][0], config['batch_size'][1]) if type(config['batch_size']) == tuple else config['batch_size']
        epochs = trial.suggest_int("epochs", config['epochs'][0], config['epochs'][1]) if type(config['epochs']) == tuple else config['epochs']
        lr = trial.suggest_float('lr', config['lr'][0], config['lr'][1]) if type(config['lr']) == tuple else config['lr']
        input_dropout = trial.suggest_float('input_dropout', config['input_dropout'][0], config['input_dropout'][1]) if type(config['input_dropout']) == tuple else config['input_dropout']
        hidden_dropout = trial.suggest_float('hidden_dropout', config['hidden_dropout'][0], config['hidden_dropout'][1]) if type(config['hidden_dropout']) == tuple else config['hidden_dropout']
        weight_decay = trial.suggest_float('weight_decay', config['weight_decay'][0], config['weight_decay'][1]) if type(config['weight_decay']) == tuple else config['weight_decay']
        l1 = trial.suggest_int("l1", config['l1'][0], config['l1'][1]) if type(config['l1']) == tuple else config['l1']
        l2 = trial.suggest_int("l2", config['l2'][0], config['l2'][1]) if type(config['l2']) == tuple else config['l2']
        optimizer_key = trial.suggest_discrete_uniform('optimizer_key',0, 1, 1) if type(config['optimizer_key']) == tuple else config['optimizer_key']
        loss_fn_key = trial.suggest_discrete_uniform('loss_fn_key',0, 1, 1) if type(config['loss_fn_key']) == tuple else config['loss_fn_key']

        if 'l3' in config.keys():
            l3 = trial.suggest_int("l3", config['l3'][0], config['l3'][1]) if type(config['l3']) == tuple else config['l3']
            model = models.DoubleLayerNet(input_dim, l1, l2, l3, output_dim, input_dropout, hidden_dropout)

        else:
            model = models.DoubleLayerNet(input_dim, l1, l2, output_dim, input_dropout, hidden_dropout)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f'Training running on {device}')

        if optimizer_key == 1:
            b1 = trial.suggest_float('b1', config['b1'][0], config['b1'][1]) if type(config['b1']) == tuple else config['b1']
            b2 = trial.suggest_float('b2', config['b2'][0], config['b2'][1]) if type(config['b2']) == tuple else config['b2']
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)

        else:
            b1, b2 = None, None
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_fn = torch.nn.L1Loss() if loss_fn_key == 0 else torch.nn.MSELoss()

        selected = {}
        for key in config.keys():
            if key in locals():
                selected[key] = locals()[key]

        with open(os.path.join(trial_dir, 'config.json'), 'w') as f:
            json.dump(selected, f)

        train_dataloader, test_dataloader, _, _ = train.load_data(features_file, (0.8,0.1,0.1), batch_size, seed, encodings)

        train_loss = np.array([])
        test_loss = np.array([])
        for epoch in range(0, epochs):
            _train_loss = train.train_loop(model, train_dataloader, loss_fn, optimizer, device)
            _test_loss = train.validation_loop(model, test_dataloader, loss_fn, device)
            print(_train_loss, _test_loss)

            train_loss = np.append(train_loss, _train_loss)
            test_loss = np.append(test_loss, _test_loss)
            trial.report(_test_loss, epoch)

            if trial.should_prune():
                _write_output_files(trial_dir, train_loss, test_loss)
                raise optuna.exceptions.TrialPruned()

        _write_output_files(trial_dir, train_loss, test_loss)
        return _test_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    importances = optuna.importance.get_param_importances(study)

    with open(os.path.join(dirname, 'importances.json'), 'w') as f:
        json.dump(importances, f)