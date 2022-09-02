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

def _write_output_files(subdir: str, train_loss: np.ndarray, test_loss: np.ndarray, trial_dict: dict):
    np.save(os.path.join(subdir, 'train_loss.npy'), train_loss)
    np.save(os.path.join(subdir, 'test_loss.npy'), test_loss)
    with open(os.path.join(subdir, 'config.json'), 'w') as f:
        json.dump(trial_dict, f)

def define_objective(path2output: str, features_file: str, config: dict):
    """
    Generates an objective function for optuna optimization.
    """

    def objective(trial):
        trial_dir = os.path.join(path2output, str(trial.number))
        os.mkdir(trial_dir)

        #generate a dictionary that contains key values sampled from the space defined in config
        trial_dict = {
        'seed': config['seed'],
        'batch_size': trial.suggest_int("batch_size", config['batch_size'][0], config['batch_size'][1]) if type(config['batch_size']) == tuple else config['batch_size'],
        'lr': trial.suggest_float('lr', config['lr'][0], config['lr'][1]) if type(config['lr']) == tuple else config['lr'],
        'encodings': config['encodings'],
        'epochs': config['epochs']
        }
        for layer_key in config.keys():
            if layer_key not in trial_dict.keys():
                layer_dict = {}
                for key in config[layer_key].keys():
                    if 'dropout' not in key:
                        layer_dict[key] = trial.suggest_int(key, config[layer_key][key][0], config[layer_key][key][1]) if type(config[layer_key][key]) == tuple else config[layer_key][key]

                    else:
                       layer_dict[key] = trial.suggest_float(key, config[layer_key][key][0], config[layer_key][key][1]) if type(config[layer_key][key]) == tuple else config[layer_key][key]
                trial_dict[layer_key] = layer_dict

        model = models.AlphafoldNet(trial_dict)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f'Training running on {device}')
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad],lr=trial_dict['lr'])   
        train_dataloader, test_dataloader, _ = train.load_data(features_file, path2output, (0.8,0.1,0.1), trial_dict['batch_size'], trial_dict['seed'], trial_dict['encodings'], False)

        train_loss = np.array([])
        test_loss = np.array([])
        model = model.float()
        for epoch in range(0, trial_dict['epochs']):
            _train_loss = train.train_loop(model, train_dataloader, loss_fn, optimizer, device)
            _test_loss = train.validation_loop(model, test_dataloader, loss_fn, device)

            train_loss = np.append(train_loss, _train_loss)
            test_loss = np.append(test_loss, _test_loss)
            trial.report(_test_loss, epoch)

            if trial.should_prune():
                _write_output_files(trial_dir, train_loss, test_loss, trial_dict)
                raise optuna.exceptions.TrialPruned()

        _write_output_files(trial_dir, train_loss, test_loss, trial_dict)
        return _test_loss

    return objective