#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Given a set of input pdb files, this script will provide predictions of the metal coordinates.
"""

#imports
import scipy
import numpy as np

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
    rmsd = np.sqrt(np.mean(np.square(solution - label)))

    return solution, rmsd