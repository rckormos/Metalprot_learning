"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for extracting binding core examples and writing their corresponding distance matrices, sequence encodings, and labels.
"""

#imports
import numpy as np
from prody import writePDB
import os
import pickle
from Metalprot_learning.core_generator import core_loader
from Metalprot_learning.core_generator import core_featurizer
from Metalprot_learning.core_generator import core_permuter
from Metalprot_learning import utils

def construct_training_example(pdb_file: str, output_dir: str, no_neighbors=1, coordinating_resis=4):
    """For a given pdb file, constructs a training example and extracts all features.

    Args:
        pdb_file (str): Path to input pdb file.
        output_dir (str): Path to output directory.
        no_neighbors (int, optional): Number of neighbors in primary sequence to coordinating residues be included in core. Defaults to 1.
        coordinating_resis (int, optional): Sets a threshold for maximum number of metal coordinating residues. Defaults to 4.
    """

    max_resis = (2*coordinating_resis*no_neighbors) + coordinating_resis
    max_permutations = int(np.prod(np.linspace(1,coordinating_resis,coordinating_resis)))

    #find all the unique cores within a given pdb structure
    cores, names = core_loader.extract_positive_cores(pdb_file, no_neighbors, coordinating_resis)
    unique_cores, unique_names = core_loader.remove_degenerate_cores(cores, names)
    if len(unique_cores) == 0:
        raise utils.NoCoresError

    #extract features for each unique core found
    completed = 0
    for core, name in zip(unique_cores, unique_names):

        full_dist_mat, binding_core_resindices, label, coords = core_featurizer.compute_distance_matrices(core, name, no_neighbors, coordinating_resis)
        if len(full_dist_mat) != max_resis * 4:
            raise utils.DistMatDimError

        if label.shape[1] != max_resis * 4:
            raise utils.LabelDimError

        encoding = core_featurizer.onehotencode(core, no_neighbors, coordinating_resis)
        if encoding.shape[1] != max_resis * 20:
            raise utils.EncodingDimError

        #permute distance matrices, labels, and encodings
        features = core_permuter.permute_features(full_dist_mat, encoding, label, binding_core_resindices)
        if len([key for key in features.keys() if type(key) == int]) > max_permutations:
            raise utils.PermutationError

        features['source'] = pdb_file

        #write files to disk
        metal_resindex = core.select(f'name {name}') .getResindices()[0]
        filename = core.getTitle() + '_' + '_'.join([str(num) for num in binding_core_resindices]) + '_' + name + str(metal_resindex)
        writePDB(os.path.join(output_dir, filename + '_core.pdb.gz'), core)
        with open(os.path.join(output_dir, filename + '_features.pkl'), 'wb') as f:
            pickle.dump(features,f)

        completed += 1

    print(f'{completed} core(s) identified and featurized for {pdb_file}')