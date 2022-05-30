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

def test_core_loader(unique_cores, unique_names, unique_noised_cores):
    dimensionality_test = len(unique_cores) == len(unique_names) == len(unique_noised_cores)
    if len(unique_cores) == 0:
        raise utils.NoCoresError

    if dimensionality_test == False:
        raise utils.CoreLoadingError

def test_featurization(full_dist_mat, noised_dist_mat, label, noised_label, encoding, max_resis):
    dist_mat_check = len(full_dist_mat) == max_resis * 4 == len(noised_dist_mat)
    label_check = label.shape[1] == max_resis * 4 == noised_label.shape[1]
    encoding_check = encoding.shape[1] == max_resis * 20

    if False in set({dist_mat_check, label_check, encoding_check}):
        raise utils.FeaturizationError

def test_permutation(observations, labels, noised_observations, noised_labels, resnums, chids, sources, max_permutations):
    dimensionality_test = len(set([len(observations), len(labels), len(noised_observations), len(noised_labels), len(resnums), len(chids), len(sources)])) == 1
    permutation_test = len(observations) == len(noised_observations) <= max_permutations
    
    if False in set({dimensionality_test, permutation_test}):
        raise utils.PermutationError

def merge_dictionaries(d1: dict, d2: dict):
    d = {}
    for k in d1.keys():
        if type(d1[k]) != list:
            placeholder = []
            placeholder.append(d1[k])

        else:
            placeholder = d1[k]

        placeholder.append(d2[k])
        d[k] = placeholder

    return d

def construct_training_example(pdb_file: str, output_dir: str, permute: bool, write: bool, no_neighbors=1, coordinating_resis=4):
    """For a given pdb file, constructs a training example and extracts all features.

    Args:
        pdb_file (str): Path to input pdb file.
        output_dir (str): Path to output directory.
        no_neighbors (int, optional): Number of neighbors in primary sequence to coordinating residues be included in core. Defaults to 1.
        coordinating_resis (int, optional): Sets a threshold for maximum number of metal coordinating residues. Defaults to 4.
    """

    max_resis = (2*coordinating_resis*no_neighbors) + coordinating_resis
    max_permutations = int(np.prod(np.linspace(1,coordinating_resis,coordinating_resis)))
    unique_cores, unique_names, unique_noised_cores = core_loader.extract_positive_cores(pdb_file, no_neighbors, coordinating_resis)
    test_core_loader(unique_cores, unique_names, unique_noised_cores)

    completed = 0
    for core, noised_core, name in zip(unique_cores, unique_noised_cores, unique_names):
        dist_mat, label, noised_dist_mat, noised_label, encoding, binding_core_resnums, binding_core_chids, metal_coord = core_featurizer.featurize(core, noised_core, name, no_neighbors, coordinating_resis)
        test_featurization(dist_mat, noised_dist_mat, label, noised_label, encoding, max_resis)

        #permute distance matrices, labels, and encodings
        if permute:
            observations, labels, noised_observations, noised_labels, resnums, chids, sources, metal_coords = core_permuter.permute_fragments(dist_mat, noised_dist_mat, encoding, label, noised_label, binding_core_resnums, binding_core_chids, pdb_file, metal_coord)
            test_permutation(observations, labels, noised_observations, noised_labels, resnums, chids, sources, max_permutations)

        else:
            observations, labels, noised_observations, noised_labels, resnums, chids, sources, metal_coords = [np.concatenate((dist_mat.flatten(), encoding.squeeze()))], [label.squeeze()], [np.concatenate((noised_dist_mat.flatten(), encoding.squeeze()))], [noised_label.squeeze()], [binding_core_resnums], [binding_core_chids], [pdb_file], [metal_coord]

        completed += 1

    if write:
        metal_chid = core.select(f'name {name}') .getChids()[0]
        metal_resnum = core.select(f'name {name}').getResnums()[0]
        filename = '_'.join([str(resnum) + str(chid) for resnum in binding_core_resnums for chid in binding_core_chids]) + f' {metal_resnum}{metal_chid}'
        writePDB(os.path.join(output_dir, filename + '_core.pdb.gz'), core)

    print(f'{completed} core(s) identified and featurized for {pdb_file}')
    return observations, labels, noised_observations, noised_labels, resnums, chids, sources, metal_coords