#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Given a set of input pdb files, this script will provide predictions of the metal coordinates.
"""

#imports
from Metalprot_learning.train import models
from Metalprot_learning.predictor import *
import os
import json
import sys
import pickle
import torch
import numpy as np

def distribute_tasks(no_jobs: int, job_id: int, pointers: list, permutations: list, observations: np.ndarray, labels: np.ndarray):
    row_indices = np.linspace(0, len(pointers), len(pointers))
    task_rows = np.array_split(row_indices, no_jobs)[job_id]
    start_ind = int(task_rows[0])
    end_ind = int(task_rows[-1]) + 1

    pointers = pointers[start_ind:end_ind]
    permutations = permutations[start_ind:end_ind]
    observations = observations[start_ind:end_ind]
    labels = labels[start_ind:end_ind]

    print(f'Predicting coordinates for indices {start_ind}:{end_ind}')

    return pointers, permutations, observations, labels

def load_data(features_file: str, arch_file: str):
    with open(arch_file, 'r') as f:
        config = json.load(f)

    with open(features_file, 'rb') as f:
        features = pickle.load(f)

    pointers = features['pointers']
    permutations = features['binding_core_identifiers']
    observations = features['observations']
    labels = features['labels']

    return config, pointers, permutations, observations, labels

if __name__ == '__main__':
    path2output = sys.argv[1] #path to store outputs  
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    #load data
    EXAMPLE = True #true if data are positive examples
    FEATURES_FILE = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/datasetV2/compiled_features.pkl'
    CONFIG_FILE = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/models/MLP_v1/23_5_2022_9_28_38_271008_741/config.json'
    WEIGHTS_FILE = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/models/MLP_v1/23_5_2022_9_28_38_271008_741/model.pth'
    config, pointers, permutations, observations, labels = load_data(FEATURES_FILE, CONFIG_FILE)
    assert len(pointers) == len(permutations) == len(observations) == len(labels)

    pointers, permutations, observations, labels = distribute_tasks(no_jobs, job_id, pointers, permutations, observations, labels)

    #load model
    model = models.DoubleLayerNet(config['input'], config['l1'], config['l2'], config['l3'], config['output'])
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location='cpu'))
    model.eval()

    #forward pass and evaulation of predictions
    predictions = model.forward(torch.from_numpy(observations)).cpu().detach().numpy()
    predicted_metal_coordinates, metal_coordinates, rmsds = predict_coordinates(predictions, pointers, permutations, example=EXAMPLE)

    if EXAMPLE:
        deviation = evaluate_positives(predicted_metal_coordinates, metal_coordinates)
        np.save(os.path.join(path2output, f'deviation{job_id}'), deviation)

    np.save(os.path.join(path2output, f'coordinates{job_id}'), predicted_metal_coordinates)
    np.save(os.path.join(path2output, f'rmsds{job_id}'), rmsds)

    failed_indices = np.argwhere(np.isnan(predicted_metal_coordinates))
    if len(failed_indices) > 0:
        failed = [pointers[int(i)] for i in failed_indices]
        with open(os.path.join(path2output, 'failed.txt'), 'a') as f:
            f.write('\n'.join(failed) + '\n')