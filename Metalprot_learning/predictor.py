"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for predicting metal coordinates.
"""

#imports
import numpy as np
import scipy
from prody import *

def triangulate(backbone_coords, distance_prediction):
    distance_prediction = distance_prediction[0:len(backbone_coords)]

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

def extract_coordinates(source_file: str, identifier_permutation, example):
    """_summary_

    Args:
        source_file (str): _description_
        positive (bool, optional): _description_. Defaults to False.
    """

    metal_coords = np.array([np.nan, np.nan, np.nan])
    core = parsePDB(source_file)
    for iteration, id in enumerate(identifier_permutation):
        residue = core.select(f'chain {id[1]}').select(f'resnum {id[0]}').select('name C CA N O').getCoords()
        coordinates = residue if iteration == 0 else np.vstack([coordinates, residue])

    if example:
        metal_coords = core.select('hetero').getCoords()
        assert len(metal_coords) == 1
        metal_coords = metal_coords[0]

    return coordinates, metal_coords

def predict_coordinates(distance_predictions, pointers, resindex_permutations, example=False):
    predicted_metal_coordinates = None
    metal_rmsds = None
    metal_coordinates = None
    completed = 0
    for distance_prediction, pointer, resindex_permutation in zip(distance_predictions, pointers, resindex_permutations):
        try:
            source_coordinates, _metal_coordinates = extract_coordinates(pointer, resindex_permutation, example)
            solution, rmsd = triangulate(source_coordinates, distance_prediction)
            completed += 1

        except:
            _metal_coordinates = np.array([np.nan, np.nan, np.nan])
            solution, rmsd = np.array([np.nan, np.nan, np.nan]), np.nan

        if type(predicted_metal_coordinates) != np.ndarray:
            predicted_metal_coordinates = solution
            metal_coordinates = _metal_coordinates
            metal_rmsds = rmsd

        else:
            predicted_metal_coordinates = np.vstack([predicted_metal_coordinates, solution])
            metal_coordinates = np.vstack([metal_coordinates, _metal_coordinates])
            metal_rmsds = np.append(metal_rmsds, rmsd)

    print(f'Coordinates and RMSDs computed for {completed} out of {len(distance_predictions)} observations.')
    return predicted_metal_coordinates, metal_coordinates, metal_rmsds

def evaluate_positives(predicted_metal_coordinates, metal_coordinates):
    deviation = np.sqrt(np.sum(np.square(metal_coordinates - predicted_metal_coordinates), axis=1))
    return deviation