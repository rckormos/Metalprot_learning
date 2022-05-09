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

def extract_coordinates(source_file: str, positive):
    """_summary_

    Args:
        source_file (str): _description_
        positive (bool, optional): _description_. Defaults to False.
    """

    core = parsePDB(source_file)
    backbone_coordinates = core.select('name C CA N O').getCoords()
    metal_coords = None

    if positive:
        metal_coords = core.select('hetero').select('name NI MN ZN CO CU MG FE').getCoords()

    return backbone_coordinates, metal_coords

def evaulate_positives(predictions, source_metal_coordinates):
    deviations = np.array([])
    for prediction, source_metal_coordinate in zip(predictions, source_metal_coordinates):
        deviation = np.linalg.norm(prediction - source_metal_coordinate)
        deviations = np.append(deviations, deviation)

    return deviations

def predict_coordinates(distance_predictions, index, positive=False):
    coordinate_lookup = {}
    for file in list(set(index)):
        coordinate_lookup[str(file)] = extract_coordinates(file, positive)

    source_coordinates = np.array([coordinate_lookup[i][0] for i in index])
    source_metal_coordinates = np.array([coordinate_lookup[i][1] for i in index])

    predictions = []
    uncertainties = []
    deviations = None
    average_deviation = None
    variance = None
    for distance_prediction, source_coordinate in zip(distance_predictions, source_coordinates):
        solution, rmsd = triangulate(source_coordinate[0], distance_prediction)
        predictions.append(solution)
        uncertainties.append(rmsd)
        
    if positive:
        deviations = evaulate_positives(predictions, source_metal_coordinates)
        average_deviation = np.mean(deviations)
        variance = np.std(deviations)

    return predictions, uncertainties, deviations, average_deviation, variance