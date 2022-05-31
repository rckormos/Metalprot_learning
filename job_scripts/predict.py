#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Given a set of input pdb files, this script will provide predictions of the metal coordinates.
"""

#imports
from Metalprot_learning.predictor import predict_coordinates
import json
import sys
import pickle
import numpy as np
import pandas as pd

def distribute_tasks(features: pd.DataFrame):

    path2output = sys.argv[1] #path to store outputs  
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    row_indices = np.linspace(0, len(features), len(features))
    task_rows = np.array_split(row_indices, no_jobs)[job_id]
    start_ind = int(task_rows[0])
    end_ind = int(task_rows[-1]) + 1

    tasks = features[start_ind:end_ind]

    print(f'Predicting coordinates for indices {start_ind}:{end_ind}')

    return path2output, job_id, tasks

def load_data(features_file: str, config_file: str):
    with open(config_file, 'r') as f:
        config = json.load(f)

    with open(features_file, 'rb') as f:
        features = pickle.load(f)

    return config, features

if __name__ == '__main__':


    #load data
    EXAMPLE = True #true if data are positive examples
    ENCODINGS = False
    FEATURES_FILE = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/datasetV4/compiled_features.pkl'
    CONFIG_FILE = '/Users/jonathanzhang/Documents/ucsf/degrado/data/metalprot_learning/models/MLP_v2/30_5_2022_14_58_0_515271_449/config.json'
    WEIGHTS_FILE = '/Users/jonathanzhang/Documents/ucsf/degrado/data/metalprot_learning/models/MLP_v2/30_5_2022_14_58_0_515271_449/model.pth'

    config, features = load_data(FEATURES_FILE, CONFIG_FILE)
    path2output, job_id, tasks = distribute_tasks(features)

    predict_coordinates(path2output, job_id, features, config, WEIGHTS_FILE, EXAMPLE, ENCODINGS)