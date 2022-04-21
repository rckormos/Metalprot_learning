#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Note that this script is meant to be called from within a SGE bash submission script. Note that you need to define
the number of jobs in this script and in the aforementioned submission script.
"""

#imports 
import os
from webbrowser import get
from Metalprot_learning.get_binding_cores import construct_training_example 

def distribute_tasks(path2examples: str, no_jobs: int, job_id: int):
    """Distributes pdb files for core generation.

    Args:
        path2examples (str): Path to directory containing examples.
        no_jobs (int): Total number of jobs.
        job_id (int): The job id.

    Returns:
        tasks (list): list of pdb files assigned to particular job id.
    """
    pdbs = [os.path.join(path2examples, file) for file in os.listdir(path2examples) if '.pdb' in file]
    tasks = [pdbs[i] for i in range(0, len(pdbs)) if i % no_jobs == job_id]

    return tasks

if __name__ == '__main__':
    path2examples = '/Users/jonathanzhang/Documents/ucsf/degrado/Metalprot_learning/data' #path to positive or negative input structures
    path2output = '/Users/jonathanzhang/Documents/ucsf/degrado/Metalprot_learning/data/outputs' #path to where you want positive

    no_jobs = 1 #set the number of jobs and job_id
    job_id = os.getenv('SGE_TASK_ID')

    tasks = distribute_tasks(path2examples, no_jobs, job_id)

    for file in tasks:
        print(file)
        construct_training_example(file, path2output) 