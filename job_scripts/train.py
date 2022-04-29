#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs model training. 
"""

#imports
from Metalprot_learning.models import *
from Metalprot_learning.datasets import *
from Metalprot_learning.trainer import *
import sys

if __name__ == '__main__':

    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    path2observations = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/observations.npy'
    path2labels = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/labels.npy'

    #define architecture of neural network
    arch = [{'input_dim': 2544, 'output_dim': 1272, 'activation': None}, 
            {'input_dim': 1272, 'output_dim': 636, 'activation': None},
            {'input_dim': 636, 'output_dim': 318, 'activation': None},
            {'input_dim': 318, 'output_dim': 48, 'activation': 'ReLU'}]


    #define hyperparameters
    epochs = 1000
    batch_size = 2000
    learning_rate = 0.001
    loss_function = 'MAE' #can be mean absolute error (MAE) or mean squared error (MSE)
    optimizer = 'SGD' #currently can only be stochastic gradient descent (SGD)
    partition = (0.8,0.1,0.1)
    seed = 42
    name = 'model1'

    model = SingleLayerNet(arch)
    train_model(model, path2output, path2observations, path2labels, epochs, batch_size, learning_rate, loss_function, 
                optimizer, name=name, partition=partition, seed=seed)