#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script optimizes model performance by tuning dropout rate. 
"""

#imports
import sys
import numpy as np
from copy import deepcopy
from Metalprot_learning.models import *
from Metalprot_learning.datasets import *
from Metalprot_learning.trainer import *

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

    #provide paths to observations and labels
    path2observations = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/observations.npy'
    path2labels = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/labels.npy'

    #define hyperparameters. 
    epochs = 2000
    batch_size = 1000
    learning_rate = 0.01
    loss_function = 'MAE' #can be mean absolute error (MAE) or mean squared error (MSE)
    optimizer = 'SGD' #currently can only be stochastic gradient descent (SGD)
    partition = (0.8,0.1,0.1)
    seed = 42

    #define dropout rate vectors
    input_layer_rates = np.linspace(.5,.9,5)
    hidden_layer_rates = np.linspace(.1,.9,9)

    #define base architecture
    arch = [{'input_dim': 2544, 'output_dim': 1272, 'activation': 'ReLU'}, 
            {'input_dim': 1272, 'output_dim': 636, 'activation': 'ReLU'},
            {'input_dim': 636, 'output_dim': 318, 'activation': 'ReLU'},
            {'input_dim': 318, 'output_dim': 48, 'activation': 'ReLU'}]

    training_data, testing_data, validation_data = split_data(path2observations, path2labels, partition, seed)
    tasks = distribute_tasks(no_jobs, job_id, arch, input_layer_rates, hidden_layer_rates)
    for architecture in tasks:
        name = os.path.join(path2output, str(architecture[0]['dropout']) + '_' + str(round(architecture[1]['dropout'],2)))
        os.makedirs(name)

        model = SingleLayerNet(architecture)
        train_model(model, name, training_data, testing_data, (epochs, batch_size, learning_rate, loss_function, optimizer))





