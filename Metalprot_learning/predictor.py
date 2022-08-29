"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for predicting metal coordinates.
"""

#imports
import os
from Metalprot_learning.train.models import SingleLayerNet, DoubleLayerNet, FourLayerNet
import pandas as pd
import torch
import numpy as np
import scipy
from prody import *

def _load_model(config: dict, WEIGHTS_FILE: str):

    if 'l3' not in config.keys():
        model = SingleLayerNet(config['input'], config['l1'], config['l2'], config['output'], config['input_dropout'], config['hidden_dropout']) 

    elif 'l4' in config.keys():
        model = FourLayerNet(config['input'], config['l1'], config['l2'], config['l3'], config['l4'],config['output'], config['input_dropout'], config['hidden_dropout'])

    else:
        model = DoubleLayerNet(config['input'], config['l1'], config['l2'], config['l3'], config['output'], config['input_dropout'], config['hidden_dropout'])

    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location='cpu'))
    model.eval()
    return model

def _triangulate(backbone_coords, distance_prediction):
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

def _extract_coordinates(source_file: str, identifier_permutation):
    """_summary_

    Args:
        source_file (str): _description_
        positive (bool, optional): _description_. Defaults to False.
    """

    core = parsePDB(source_file)
    for iteration, id in enumerate(identifier_permutation):
        residue = core.select(f'chain {id[1]}').select(f'resnum {id[0]}').select('name C CA N O').getCoords()
        coordinates = residue if iteration == 0 else np.vstack([coordinates, residue])

    return coordinates

def predict_coordinates(path2output: str, job_id: int, features: pd.DataFrame, config: dict, weights_file: str, example: bool, encodings: bool):

    model = _load_model(config, weights_file)
    X = np.hstack([np.vstack([array for array in features['distance_matrices']]), np.vstack([array for array in features['encodings']])]) if encodings else np.vstack([array for array in features['distance_matrices']])
    prediction = model.forward(torch.from_numpy(X)).cpu().detach().numpy()

    deviation = np.array([np.nan] * len(prediction))
    completed = 0
    for distance_prediction, pointer, resindex_permutation in zip(prediction, list(features['source']), list(features['identifiers'])):
        try:
            source_coordinates = _extract_coordinates(pointer, resindex_permutation)
            solution, rmsd = _triangulate(source_coordinates, distance_prediction)
            completed += 1

        except:
            solution, rmsd = np.array([np.nan, np.nan, np.nan]), np.nan

        if 'solutions' not in locals():
            solutions = solution
            rmsds = rmsd

        else:
            solutions = np.vstack([solutions, solution])
            rmsds = np.append(rmsds, rmsd)

    if example:
        ground_truth = np.vstack(list(features['metal_coords']))
        deviation = np.sqrt(np.sum(np.square(ground_truth - solutions), axis=1))

    predictions = pd.DataFrame({'predicted_distances': list(prediction),
        'predicted_coordinates': list(solutions),
        'confidence': rmsds,
        'deviation': deviation,
        'barcodes': features['barcode'].to_numpy(),
        'sources': list(features['source']),
        'identifiers': list(features['identifiers'])})

    predictions.to_pickle(os.path.join(path2output, f'predictions{job_id}.pkl'))

    failed_indices = np.argwhere(np.isnan(solutions)).squeeze()
    if len(failed_indices) > 0:
        failed = [list(features['source'])[int(i)] for i in failed_indices]
        with open(os.path.join(path2output, 'failed.txt'), 'a') as f:
            f.write('\n'.join(failed) + '\n')

    print(f'Coordinates and RMSDs computed for {completed} out of {len(prediction)} observations.')