#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Given a set of input pdb files, this script will provide predictions of the metal coordinates.
"""

#imports
from Metalprot_learning.trainer.models import SingleLayerNet
from Metalprot_learning.predictor import *
import os
import json
import sys
import pickle
import torch
import numpy as np

def distribute_tasks(path2pdbs: str, no_jobs: int, job_id: int):
    """Distributes pdb files for core generation.

    Args:
        path2pdbs (str): Path to directory containing pdb files.
        no_jobs (int): Total number of jobs.
        job_id (int): The job id.

    Returns:
        tasks (list): list of pdb files assigned to particular job id.
    """
    pdbs = [os.path.join(path2pdbs, file) for file in os.listdir(path2pdbs) if '.pdb' in file]
    tasks = [pdbs[i] for i in range(0, len(pdbs)) if i % no_jobs == job_id]

    return tasks

def load_data(features_file: str, weights_file: str, arch_file: str):
    with open(arch_file, 'r') as f:
        arch = json.load(f)

    with open(features_file, 'rb') as f:
        features = pickle.load(f)

    pointers = features['pointers']
    permutations = features['permutations']
    observations = features['observations']
    labels = features['labels']

    return arch, pointers, permutations, observations, labels

if __name__ == '__main__':
    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    #load data
    examples = True #true if data are positive examples
    features_file = '/Users/jonathanzhang/Documents/ucsf/degrado/data/metalprot_learning/ZN_binding_cores/datasetV1/compiled_features.pkl'
    arch_file = '/Users/jonathanzhang/Documents/ucsf/degrado/data/metalprot_learning/models/MLP_v1/2001_1000_0.01_MAE_SGD/architecture.json'
    weights_file = '/Users/jonathanzhang/Documents/ucsf/degrado/data/metalprot_learning/models/MLP_v1/2001_1000_0.01_MAE_SGD/model.pth'
    arch, pointers, permutations, observations, labels = load_data(features_file, weights_file, arch_file)

    #load model
    model = SingleLayerNet(arch)
    model.load_state_dict(torch.load(weights_file))
    model.eval()

    #forward pass and evaulation of predictions
    predictions = model.forward(torch.from_numpy(observations)).cpu().detach().numpy()
    predicted_metal_coordinates, rmsds = predict_coordinates(predictions, pointers, permutations)

    if examples:
        deviation = evaluate_positives(predicted_metal_coordinates, pointers)
        np.sav(os.path.join(path2output, 'deviation'))

    np.save(os.path.join(path2output, 'coordinates'))
    np.save(os.path.join(path2output, 'rmdsds'))
