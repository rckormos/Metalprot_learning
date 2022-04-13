#!/bin/usr/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Note that this script is meant to be called from within a SGE bash submission script. Note that you need to define
the number of jobs in this script and in the aforementioned submission script.
"""

#imports 
import os
from src.extractor.get_binding_cores import extract_cores

def distribute_tasks(path2positives: str, path2negatives: str, no_jobs: int, job_id: int):
    """Distributes pdb files for core generation.

    Args:
        path2positives (str): Path to directory containing positive examples.
        path2negatives (str): Path to directory containing negative examples.
        no_jobs (int): Total number of jobs.
        job_id (int): The job id.

    Returns:
        _type_: _description_
    """
    positive_pdbs = [file for file in os.listdir(path2positives) if '.pdb' in file]
    negative_pdbs = [file for file in os.listidr(path2negatives) if '.pbd' in file]

    positive_tasks = [positive_pdbs[i] for i in range(0, len(positive_pdbs)) if i % no_jobs == job_id]
    negative_tasks = [negative_pdbs[i] for i in range(0, len(negative_pdbs)) if i % no_jobs == job_id]
    return positive_tasks, negative_tasks

if __name__ == '__main__':
    path2positives = '' #path to positive or negative input structures
    path2negatives = ''

    path2positive_cores = '' #path to where you want positive/negative core files dumped to
    path2negative_cores = ''

    no_jobs = 100 #number of jobs you are submitting

    try: 
        job_id = os.environ['SGE_TASK_ID'] - 1
    except:
        job_id = 0

    positive_tasks, negative_tasks = distribute_tasks(path2positives, path2negatives, no_jobs, job_id)

    for positive_file in positive_tasks:
        _ = extract_cores(positive_file, path2positive_cores, True)

    for negative_file in negative_tasks:
        _ = extract_cores(negative_file, path2negative_cores)