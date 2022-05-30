"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for extracting binding core examples and writing their corresponding distance matrices, sequence encodings, and labels.
"""

#imports
import numpy as np
from prody import writePDB
import os
import pickle
from Metalprot_learning.core_generator import core_loader, core_featurizer, core_permuter
from Metalprot_learning import utils

def test_core_loader(unique_cores, unique_names):
    dimensionality_test = len(unique_cores) == len(unique_names)
    if len(unique_cores) == 0:
        raise utils.NoCoresError

    if dimensionality_test == False:
        raise utils.CoreLoadingError

def test_featurization(full_dist_mat, label, encoding, max_resis):
    dist_mat_check = len(full_dist_mat) == max_resis * 4
    label_check = label.shape[1] == max_resis * 4
    encoding_check = encoding.shape[1] == max_resis * 20

    if False in set({dist_mat_check, label_check, encoding_check}):
        raise utils.FeaturizationError

def test_permutation(features, max_permutations):
    dimensionality_test = len(set([len(features[key]) for key in features.keys()])) == 1
    permutation_test = len(features['observations']) <= max_permutations
    
    if False in set({dimensionality_test, permutation_test}):
        raise utils.PermutationError

def construct_training_example(pdb_file: str, output_dir: str, permute: bool, no_neighbors=1, coordinating_resis=4):
    """For a given pdb file, constructs a training example and extracts all features.

    Args:
        pdb_file (str): Path to input pdb file.
        output_dir (str): Path to output directory.
        no_neighbors (int, optional): Number of neighbors in primary sequence to coordinating residues be included in core. Defaults to 1.
        coordinating_resis (int, optional): Sets  a threshold for maximum number of metal coordinating residues. Defaults to 4.
    """

    max_resis = (2*coordinating_resis*no_neighbors) + coordinating_resis
    max_permutations = int(np.prod(np.linspace(1,coordinating_resis,coordinating_resis)))

    #find all the unique cores within a given pdb structure
    cores, names = core_loader.extract_positive_cores(pdb_file, no_neighbors, coordinating_resis)
    unique_cores, unique_names = core_loader.remove_degenerate_cores(cores, names)
    test_core_loader(unique_cores, unique_names)

    #extract features for each unique core found
    completed = 0
    for core, name in zip(unique_cores, unique_names):

        full_dist_mat, binding_core_identifiers, label, metal_coords = core_featurizer.compute_distance_matrices(core, name, no_neighbors, coordinating_resis)
        encoding = core_featurizer.onehotencode(core, no_neighbors, coordinating_resis)
        test_featurization(full_dist_mat, label, encoding, max_resis)

        #permute distance matrices, labels, and encodings
        if permute:
            features = core_permuter.permute_fragments(full_dist_mat, encoding, label, binding_core_identifiers)
            test_permutation(features, max_permutations)

        else:
            features = {'observations': [np.concatenate((full_dist_mat.flatten(), encoding.squeeze()))], 'labels': [label.squeeze()], 'identifiers': [binding_core_identifiers]}

        #write files to disk
        metal_chid = core.select(f'name {name}') .getChids()[0]
        metal_resnum = core.select(f'name {name}').getResnums()[0]
        filename = core.getTitle() + '_' + '_'.join([str(tup[0]) + tup[1] for tup in binding_core_identifiers]) + '_' + name + str(metal_resnum) + metal_chid
        features['source'] = [os.path.join(output_dir, filename + '_core.pdb.gz')] * len(features['observations'])
        features['metal_coords'] = [metal_coords] * len(features['observations'])

        writePDB(os.path.join(output_dir, filename + '_core.pdb.gz'), core)
        with open(os.path.join(output_dir, filename + '_features.pkl'), 'wb') as f:
            pickle.dump(features,f)

        completed += 1

    print(f'{completed} core(s) identified and featurized for {pdb_file}')