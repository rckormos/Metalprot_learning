"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script contains unit tests for functions in process.py
"""

#imports 
import numpy as np
import pickle
import os
import sys
from prody import *

def load_data():
    "Helper function for loading core files"
    data_path = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores'

    pdbs = [os.path.join(data_path, file) for file in os.listdir(data_path) if '.pdb.gz' in file]

    compiled_observations_file = os.path.join(data_path, 'observations.npy')
    compiled_labels_file = os.path.join(data_path, 'labels.npy')
    index_file = os.path.join(data_path, 'index.pkl')

    compiled_observations = np.load(compiled_observations_file)
    compiled_labels = np.load(compiled_labels_file)
    with open(index_file, 'rb') as f:
        index = pickle.load(f)

    return pdbs, compiled_observations, compiled_labels, index

def test_process():
    pdbs, compiled_observations, compiled_labels, index = load_data()
    ids = index['ids']
    coords = index['coordinates']
    metals = index['metals']

    #check that lengths are the same
    assert len(compiled_observations) == len(compiled_labels) == len(ids) == len(coords) == len(metals)

    for row_ind in range(0, len(compiled_observations)):
        observation = compiled_observations[row_ind]
        label = compiled_labels[row_ind]
        id = ids[row_ind]
        coord = coords[row_ind]
        metal = coords[row_ind]
        tol = 10e-6

        ref_pdb_file = list(filter(lambda x: id in x, pdbs))[0]
        ref_pdb = parsePDB(ref_pdb_file)

        ref_coord = ref_pdb.select('hetero').select(f'name {metal}').getCoords()[0]
        assert np.linalg.norm(ref_coord - coord) < tol

        ref_name = ref_pdb.select('hetero').select(f'name {metal}').getNames()
        assert metal == ref_name



