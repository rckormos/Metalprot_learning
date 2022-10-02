#!/usr/bin/env python3
"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>
Script for running metal binding site prediction pipeline.
"""

#imports
import os
import sys
import torch
import numpy as np
import pandas as pd
from Metalprot_learning import loader

def distribute_tasks(path2pdbs: str, no_jobs: int, job_id: int):
    """
    Distributes pdb files across multiple cores for loading.
    """
    pdbs = [os.path.join(path2pdbs, file) for file in os.listdir(path2pdbs) if '.pdb' in file]
    tasks = [pdbs[i] for i in range(0, len(pdbs)) if i % no_jobs == job_id]
    return tasks

def instantiate_models():
    classifier = None
    regressor = None
    return classifier, regressor

if __name__ == '__main__':
    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    PATH2PDBS = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/src'
    tasks = distribute_tasks(PATH2PDBS, no_jobs, job_id)
    sources, identifiers, features = [], [], []
    failed = []
    df = pd.DataFrame()
    for pdb_file in tasks:
        try:
            protein = loader.Protein(pdb_file)
            cores = protein.get_putative_cores()
            identifiers = [core.identifiers for core in cores]
            df = pd.concat([df, pd.DataFrame({'cores': identifiers, 'source': [pdb_file] * len(identifiers)})])

        except:
            failed.append(pdb_file)

    df.to_pickle(os.path.join(path2output, f'core_df{job_id}.pkl'))
    failed = list(filter(None, failed))
    if len(failed) > 0:
        with open(os.path.join(path2output, 'failed.txt'), 'a') as f:
            f.write('\n'.join([line for line in failed]) + '\n')

        _features = np.stack([core.compute_channels() for core in cores], axis=0)
        sources += [pdb_file] * len(cores)
        identifiers += [core.identifiers for core in cores]
        features.append(_features)
        
    features = np.stack(features, axis=0)
    identifiers = np.array(identifiers)
    sources = np.array(sources)
    
    classifier, regressor = instantiate_models()
    classifications = classifier.forward(torch.from_numpy(features)).cpu().detach().numpy().round()
    metal_sites_inds = np.argwhere(classifications == 1).flatten()

    regressions = regressor.forward(torch.from_numpy(features[metal_site_inds])).cpu().detach().numpy().round()
    #triangulate to get coords and confidences


    

    

