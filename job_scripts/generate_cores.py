#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Note that this script is meant to be called from within a SGE bash submission script. 
"""

#imports 
import os
import sys
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
    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1
    
    #YOU WILL NEED TO EDIT THIS PATH HERE
    path2examples = '/wynton/home/rotation/jzhang1198/protein_des_mod/Metalprot_learning/data'

    failed = []
    tasks = distribute_tasks(path2examples, no_jobs, job_id)
    for file in tasks:
        try:
            print(f'Current file: {file}')
            construct_training_example(file, path2output)
            print('')

        except:
            failed.append(file)

    if len(failed) != 0:
        with open(os.path.join(path2output, 'failed.txt'), 'w') as f:
            f.write('\n'.join([path for path in failed]))

