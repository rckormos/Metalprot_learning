"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for hyperparameter tuning.
"""

#imports
import os
import json
import torch
import optuna
import numpy as np
from Metalprot_learning.train import models
from Metalprot_learning.train import train

def _write_output_files(subdir: str, selected: dict, model, train_loss: np.ndarray, test_loss: np.ndarray):
    np.save(os.path.join(subdir, 'train_loss.npy'), train_loss)
    np.save(os.path.join(subdir, 'test_loss.npy'), test_loss)

    with open(os.path.join(subdir, 'config.json'), 'w') as f:
        json.dump(selected, f)

    torch.save(model.state_dict, os.path.join(subdir, 'model.pth'))

def define_objective(path2output: str, features_file: str, config: dict, encodings: bool):

    input_dim = 2544 if encodings else 1
    output_dim = 48

    def objective(trial):
        trial_dir = os.path.join(path2output, str(trial.number))
        os.mkdir(trial_dir)

        seed = np.random.randint(0,1000)
        batch_size = trial.suggest_int("batch_size", config['batch_size'][0], config['batch_size'][1]) if len(config['batch_size']) == 2 else config['batch_size']
        epochs = trial.suggest_int("epochs", config['epochs'][0], config['epochs'][1]) if len(config['epochs']) == 2 else config['epochs']
        lr = trial.suggest_float('lr', config['lr'][0], config['lr'][1]) if len(config['lr']) == 2 else config['lr']
        input_dropout = trial.suggest_float('input_dropout', config['input_dropout'][0], config['input_dropout'][1]) if len(config['input_dropout']) == 2 else config['input_dropout']
        hidden_dropout = trial.suggest_float('hidden_dropout', config['hidden_dropout'][0], config['hidden_dropout'][1]) if len(config['hidden_dropout']) == 2 else config['hidden_dropout']
        alpha = trial.suggest_float('weight_decay', config['weight_decay'][0], config['weight_decay'][1]) if len(config['weight_decay']) == 2 else config['weight_decay']
        l1 = trial.suggest_int("l1", config['l1'][0], config['l1'][1]) if len(config['l1']) == 2 else config['l1']
        l2 = trial.suggest_int("l2", config['l2'][0], config['l2'][1]) if len(config['l2']) == 2 else config['l2']

        if 'l3' in config.keys():
            l3 = trial.suggest_int("l3", config['l3'][0], config['l3'][1]) if len(config['l3']) == 2 else config['l3']
            model = models.DoubleLayerNet(input_dim, l1, l2, l3, output_dim, input_dropout, hidden_dropout)
            selected = {'seed': seed, 'batch_size': batch_size, 'lr': lr, 'input_dropout': input_dropout, 'hidden_dropout': hidden_dropout, 'weight_decay': alpha, 'l1': l1, 'l2': l2, 'l3': l3}

        else:
            model = models.DoubleLayerNet(input_dim, l1, l2, output_dim, input_dropout, hidden_dropout)
            selected = {'seed': seed, 'batch_size': batch_size, 'lr': lr, 'input_dropout': input_dropout, 'hidden_dropout': hidden_dropout, 'weight_decay': alpha, 'l1': l1, 'l2': l2}

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f'Training running on {device}')

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=alpha)
        loss_fn = torch.nn.L1Loss()

        train_dataloader, test_dataloader, _ = train.load_data(features_file, (0.8,0.1,0.1), batch_size, seed, False)

        train_loss = np.array([])
        test_loss = np.array([])
        for epoch in range(0, epochs):
            _train_loss = train.train_loop(model, train_dataloader, loss_fn, optimizer, device)
            _test_loss = train.validation_loop(model, test_dataloader, loss_fn, device)

            train_loss = np.append(train_loss, _train_loss)
            test_loss = np.append(test_loss, _test_loss)
            trial.report(_test_loss, epoch)

            if trial.should_prune():
                _write_output_files(trial_dir, selected, model, train_loss, test_loss)
                raise optuna.exceptions.TrialPruned()

        _write_output_files(trial_dir, selected, model, train_loss, test_loss)
        return _test_loss

    return objective