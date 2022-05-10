"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for predicting metal coordinates.
"""

#import
import numpy as np
import scipy
from prody import *

def triangulate(backbone_coords, distance_prediction):

    guess = backbone_coords[0]
    def objective(v):
        x,y,z = v
        distances = np.zeros(backbone_coords.shape[0])
        for i in range(0, backbone_coords.shape[0]):
            atom = backbone_coords[i]
            dist = np.linalg.norm(atom - np.array([x,y,z]))
            distances[i] = dist
        rmsd = np.sqrt(np.mean(np.square(distances - distance_prediction)))
        return rmsd
    
    result = scipy.optimize.minimize(objective, guess)
    solution = result.x
    rmsd = objective(solution)

    return solution, rmsd

def extract_coordinates(source_file: str, resnum_permutation):
    """_summary_

    Args:
        source_file (str): _description_
        positive (bool, optional): _description_. Defaults to False.
    """

    core = parsePDB(source_file)
    for iteration, resnum in enumerate(resnum_permutation):
        residue = core.select(f'resnum {resnum}').select('name C CA N O').getCoords()

        if iteration == 0:
            coordinates = residue
        else:
            coordinates = np.vstack([coordinates, residue])

    return coordinates

def predict_coordinates(distance_predictions, pointers, resnum_permutations):

    predicted_metal_coordinates = None
    metal_rmsds = None
    for distance_prediction, pointer, resnum_permutation in zip(distance_predictions, pointers, resnum_permutations):
        source_coordinates = extract_coordinates(pointer, resnum_permutation)
        solution, rmsd = triangulate(source_coordinates, distance_prediction)

        if not predicted_metal_coordinates:
            predicted_metal_coordinates = solution
            metal_rmsds = rmsd

        else:
            predicted_metal_coordinates = np.vstack([predicted_metal_coordinates, solution])
            metal_rmsds = np.append(metal_rmsds, rmsd)

    return predicted_metal_coordinates, metal_rmsds

def evaulate_positives(predicted_metal_coordinates, pointers):
    metal_coordinate_lookup = {}
    for file in pointers:
        core = parsePDB(file)
        metal_coordinate_lookup[file] = core.select('hetero').select('name NI MN ZN CO CU MG FE').getCoords()

    source_metal_coordinates = np.array([metal_coordinate_lookup[i] for i in pointers])
    deviation = np.sqrt(np.sum(np.square(source_metal_coordinates - predicted_metal_coordinates), axis=1))
    
    return deviation

