#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Given a set of input pdb files, this script will provide predictions of the metal coordinates.
"""

#imports
import scipy
import numpy as np
import torch
import sys
import json
from Metalprot_learning.core_generator import *
from Metalprot_learning.models import *

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

def compute_features_matrix(pdbs: list, no_neighbors: int, coordinating_resis: int):
    """Writes features matrix.

    Args:
        pdbs (list): List of paths to input pdb files.
        no_neighbors (int): Number of adjacent neighbors to coordinating resis in primary sequence.
        coordinating_resis (int): Upper limit on number of coordinating residues considered.

    Returns:
        features (np.ndarray): Features matrix with rows indexed by observations.
        resnums (list): List of resnums for each observation. Indexed by observation.
        names (list): List of metal identities, if known. Indexed by observation. 
        core_list (list): List of cores.
    """

    metal_names = []
    resnums = []
    core_list = []
    ids = []
    no_iter = 0
    for pdb in pdbs:
        cores, names = extract_cores(pdb, no_neighbors, coordinating_resis)
        id = pdb.split('/')[-1].split('.')[0]

        for core, name in zip(cores, names):
            full_dist_mat, binding_core_resnums = compute_distance_matrices(core, no_neighbors, coordinating_resis)
            encoding = onehotencode(core, no_neighbors, coordinating_resis)

            if no_iter == 0:
                features = np.concatenate((full_dist_mat.flatten(), encoding.squeeze()))

            else:
                features = np.vstack([features, np.concatenate((full_dist_mat.flatten(), encoding.squeeze()))])

            metal_names.append(name)
            resnums.append(binding_core_resnums)
            core_list.append(core)
            ids.append(id)
            no_iter += 1
    
    return features, resnums, metal_names, core_list, ids

def triangulate(core, resnums, label):
    """Given core coordinates and a label, will compute putative coordinates of metal and uncertainty. 

    Args:
        core (_type_): _description_
        resnums (_type_): _description_
        label (_type_): _description_

    Returns:
        solution (np.array): x,y,z coordinates of metal.
        rmsd (float): Uncertainty metric of solution.
    """
    for i, resnum in enumerate(resnums):
        if i == 0:
            coords = core.select(f'resnum {resnum}').select('name C CA N O').getCoords()
            guess = coords[0]
        else:
            coords = np.vstack([coords, core.select(f'resnum {resnum}').select('name C CA N O').getCoords()])
    

    def objective(v):
        x,y,z = v
        distances = np.zeros(coords.shape[0])
        for i in range(0, coords.shape[0]):
            atom = coords[i]
            dist = np.linalg.norm(atom - np.array([x,y,z]))
            distances[i] = dist
        rmsd = np.sqrt(np.mean(np.square(distances - label)))
        return rmsd
    
    result = scipy.optimize.minimize(objective, guess)
    solution = result.x
    rmsd = objective(solution)

    return solution, rmsd

if __name__ == '__main__':
    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    no_neighbors = 1
    coordinating_resis = 4
    path2pdbs = ''
    tasks = distribute_tasks(path2pdbs, no_jobs, job_id)

    path2model = ''
    path2architecture = ''
    with open(path2architecture, 'r') as f:
        arch = json.load(f)
    model = SingleLayerNet(arch)
    model.load_state_dict(torch.load(path2model))
    
    features, resnums, names, cores, ids = compute_features_matrix(tasks, no_neighbors, coordinating_resis)
    distances = model(torch.from_numpy(features)).cpu().detach().numpy()

    file_contents = []
    for i in range(0, distances.shape[0]):
        resnum = resnums[i]
        distance = distances[i][0:len(resnum)*4]
        name = names[i]
        core = cores[i]
        id = ids[i]
        solution, uncertainty = triangulate(core, resnum, distance)
        line = str(id) + '  ' + str(name) + '  ' + str(solution) + '  ' + str(uncertainty) + '  ' + str(resnums)
        file_contents.append(line)

    with open(os.path.join(path2output, 'predicted_distances.txt'), 'w') as f:
        f.write('\n'.join([line for line in file_contents]))
