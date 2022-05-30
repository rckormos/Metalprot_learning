#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Compiles training examples into an observation and label matrix.
"""

#imports
import os
import sys
from Metalprot_learning.processor import compile_data

def distribute_tasks(path2features: str, no_jobs: int, job_id: int):
    """Distributes pdb files for core generation.

    Args:
        path2features (str): Path to directory containing feature files.
        no_jobs (int): Total number of jobs.
        job_id (int): The job id.

    Returns:
        tasks (list): list of feature files assigned to particular job id.
    """
    feature_files = [os.path.join(path2features, file) for file in os.listdir(path2features) if 'features.pkl' in file]
    tasks = [feature_files[i] for i in range(0, len(feature_files)) if i % no_jobs == job_id]

    return tasks

if __name__ == '__main__':
    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    PATH2FEATURES = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/datasetV3'
    PERMUTED = True

    feature_files = distribute_tasks(PATH2FEATURES, no_jobs, job_id)
    compile_data(PATH2FEATURES, job_id, feature_files, PERMUTED)