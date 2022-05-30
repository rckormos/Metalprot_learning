#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Note that this script is meant to be called from within a SGE bash submission script. Variables that need to be defined by user are in all caps.
"""

#imports 
import pickle
import os
import sys
from Metalprot_learning.core_generator.core_creator import construct_training_example
from Metalprot_learning import utils

def distribute_tasks(path2examples: str):
    """Distributes pdb files for core generation.

    Args:
        path2examples (str): Path to directory containing examples.
        no_jobs (int): Total number of jobs.
        job_id (int): The job id.

    Returns:
        tasks (list): list of pdb files assigned to particular job id.
    """

    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    pdbs = [os.path.join(path2examples, file) for file in os.listdir(path2examples) if '.pdb' in file]
    tasks = [pdbs[i] for i in range(0, len(pdbs)) if i % no_jobs == job_id]
    return path2output, tasks, job_id

def run_construct_training_example(file: str, path2output: str, permute: bool, write: bool, no_neighbors: int, coordination_number: int):
    """Calls main function from get_binding_cores.py to construct training example. For quality control and de-bugging purposes, there are multiple try/except statements.

    Args:
        file (str): Path to pdb file.
        path2output (str): Path to output.

    Returns:
        failed_file_line: Line to write to failed.txt
    """

    # try:
    _features = construct_training_example(file, path2output, permute=PERMUTE, write=WRITE, no_neighbors=NEIGHBORS, coordinating_resis=COORDINATION_NUMBER)
    failed_file_line = None

    # except utils.NoCoresError as e:
    #     _features = None
    #     failed_file_line = file + ' No cores identified'

    # except utils.AlignmentError as e:
    #     _features = None
    #     failed_file_line = file + ' Error identifying unique cores'

    # except utils.CoreLoadingError as e:
    #     _features = None
    #     failed_file_line = file + ' Error loading cores'

    # except utils.FeaturizationError as e:
    #     _features = None
    #     failed_file_line = file + ' Error featurizing cores'

    # except utils.EncodingError as e:
    #     _features = None
    #     failed_file_line = file + ' Unrecognized amino acid during sequence encoding'

    # except utils.PermutationError as e:
    #     _features = None
    #     failed_file_line = file + ' Issue with permutation of fragments'

    # except:
    #     _features = None
    #     failed_file_line = file + ' Unknown error occured'

    return _features, failed_file_line

def merge_dictionaries(d1: dict, d2: dict):
    d = {}
    for k in d1.keys():
        if type(d1[k]) != list:
            placeholder = []
            placeholder.append(d1[k])

        else:
            placeholder = d1[k]

        placeholder.append(d2[k])
        d[k] = placeholder

    return d

def write_failed_file(path2output: str, failed: list):
    failed = list(filter(None, failed))
    if len(failed) > 0:
        with open(os.path.join(path2output, 'failed.txt'), 'a') as f:
            f.write('\n'.join([line for line in failed]) + '\n')

if __name__ == '__main__':
    
    PATH2EXAMPLES = '/Users/jonathanzhang/Documents/ucsf/degrado/data/metalprot_learning/ZN_binding_cores/src'
    PERMUTE = True
    WRITE = False
    NEIGHBORS = 1
    COORDINATION_NUMBER = 4

    path2output, tasks, job_id = distribute_tasks(PATH2EXAMPLES)

    failed = []
    for file in tasks:
        print(file)
        _features, failed_file_line = run_construct_training_example(file, path2output, PERMUTE, WRITE, NEIGHBORS, COORDINATION_NUMBER)
        failed.append(failed_file_line)

        if _features:
            _observations, _labels, _noised_observations, _noised_labels, _resnums, _chids, _sources, _metal_coords = _features

            if 'observations' not in locals():
                observations, labels, noised_observations, noised_labels, resnums, chids, sources, metal_coords = _observations, _labels, _noised_observations, _noised_labels, _resnums, _chids, _sources, _metal_coords

            else:
                observations += _observations
                labels += _labels
                noised_observations += _noised_observations
                noised_labels += _noised_labels
                resnums += _resnums
                chids += _chids
                sources += _sources
                metal_coords += _metal_coords

        else:
            continue

    features = {'observations': observations,
        'labels': labels,
        'noised_observations': noised_observations,
        'noised_labels': noised_labels,
        'resnums': resnums,
        'chids': chids,
        'sources': sources,
        'coords': metal_coords}

    with open(os.path.join(path2output, f'features{job_id}.pkl'), 'wb') as f:
        pickle.dump(features, f)

    write_failed_file(path2output, failed)