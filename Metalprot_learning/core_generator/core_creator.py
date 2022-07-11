"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for extracting binding core examples and writing their corresponding distance matrices, sequence encodings, and labels.
"""

#imports
import numpy as np
from prody import writePDB
import os
import pickle
from Metalprot_learning.core_generator import core_loader, core_featurizer, core_permuter, core_noiser
from Metalprot_learning.core_generator.core_permuter import _trim
from Metalprot_learning import utils

def test_core_loader(unique_cores, unique_names, unique_numbers):
    dimensionality_test = len(unique_cores) == len(unique_names) == len(unique_numbers)
    if len(unique_cores) == 0:
        raise utils.NoCoresError

    if dimensionality_test == False:
        raise utils.CoreLoadingError

def test_featurization(full_dist_mat, label, noised_dist_mat, noised_label, encoding, max_resis, c_beta, trim):

    if trim:
        length = 1170 if c_beta else 1128
        dist_mat_check = len(full_dist_mat) == len(noised_dist_mat) == length

    else:
        max_atoms = max_resis * 5 if c_beta else max_resis * 4
        dist_mat_check = len(full_dist_mat) == len(noised_dist_mat) == max_atoms 
        
    label_check = label.shape[1] ==  noised_label.shape[1] == max_atoms
    encoding_check = encoding.shape[1] == max_resis * 20

    if False in set({dist_mat_check, label_check, encoding_check}):
        raise utils.FeaturizationError

def test_noiser(noised_cores, unique_cores):
    for n, c in zip(noised_cores, unique_cores):
        resnum_check = set([bool(i==j) for i,j in zip(n.getResnums(), c.select('protein and name CA N C O CB').getResnums())])
        resname_check = set([bool(i==j) for i,j in zip(n.getResnames(), c.select('protein and name CA N C O CB').getResnames())])
        name_check = set([bool(i==j) for i,j in zip(n.getNames(), c.select('protein and name CA N C O CB').getNames())])

        if False in resnum_check or False in name_check or False in resname_check:
            raise utils.NoisingError

def test_permutation(features, max_permutations):
    dimensionality_test = len(set([len(features[key]) for key in features.keys()])) == 1
    permutation_test = len(features['distance_matrices']) <= max_permutations
    
    if False in set({dimensionality_test, permutation_test}):
        raise utils.PermutationError

def construct_training_example(pdb_file: str, output_dir: str, permute: bool, c_beta: bool, trim: bool, coordinating_resis=(2,4), no_neighbors=1):
    """For a given pdb file, constructs a training example and extracts all features.

    Args:
        pdb_file (str): Path to input pdb file.
        output_dir (str): Path to output directory.
        no_neighbors (int, optional): Number of neighbors in primary sequence to coordinating residues be included in core. Defaults to 1.
        coordinating_resis (int, optional): Sets  a threshold for maximum number of metal coordinating residues. Defaults to 4.
    """

    max_resis = (2*coordinating_resis[1]*no_neighbors) + coordinating_resis[1]
    max_permutations = int(np.prod(np.linspace(1,coordinating_resis[1],coordinating_resis[1])))

    #find all the unique cores within a given pdb structure
    cores, names, coordinating_redindices = core_loader.extract_positive_cores(pdb_file, no_neighbors, coordinating_resis)
    unique_cores, unique_names, unique_coordinating_resindices = core_loader.remove_degenerate_cores(cores, names, coordinating_redindices)
    test_core_loader(unique_cores, unique_names, unique_coordinating_resindices)

    #add coordinate noise to cores
    noised_cores = [core_noiser.apply_noise_gaussian(unique_cores[i]) for i in range(0,len(unique_cores))]
    test_noiser(noised_cores, unique_cores)

    #extract features for each unique core found
    completed = 0
    for core, noised_core, name, number in zip(unique_cores, noised_cores, unique_names, unique_coordinating_resindices):

        full_dist_mat, label, noised_dist_mat, noised_label, metal_coords, binding_core_identifiers = core_featurizer.compute_distance_matrices(core, noised_core,  name, no_neighbors, coordinating_resis[1], c_beta)
        encoding = core_featurizer.onehotencode(core, no_neighbors, coordinating_resis[1])
        coordinate_label = core_featurizer.compute_coordinate_label(core, number, no_neighbors, coordinating_resis[1])
        test_featurization(full_dist_mat, label, noised_dist_mat, noised_label, encoding, max_resis, c_beta)

        #permute distance matrices, labels, and encodings
        if permute:
            features = core_permuter.permute_fragments(full_dist_mat, label, noised_dist_mat, noised_label, encoding, binding_core_identifiers, coordinate_label, c_beta, trim)
            test_permutation(features, max_permutations)

        else:
            full_dist_mat, noised_dist_mat = (_trim(full_dist_mat), _trim(noised_dist_mat)) if trim else (full_dist_mat.flatten().squeeze(), noised_dist_mat.flatten().squeeze())
            features = {'distance_matrices': [full_dist_mat], 'noised_distance_matrices': [noised_dist_mat],'labels': [label.squeeze()], 
            'noised_label': [noised_label.squeeze()],'identifiers': [binding_core_identifiers], 'encodings': [encoding.squeeze()], 'coordinate_labels': [coordinate_label]}

        #write files to disk
        metal_chid = core.select(f'name {name}') .getChids()[0]
        metal_resnum = core.select(f'name {name}').getResnums()[0]
        filename = core.getTitle() + '_' + '_'.join([str(tup[0]) + tup[1] for tup in binding_core_identifiers]) + '_' + name + str(metal_resnum) + metal_chid
        features['source'] = [os.path.join(output_dir, filename + '_core.pdb.gz')] * len(features['distance_matrices'])
        features['metal_coords'] = [metal_coords] * len(features['distance_matrices'])
        features['metal_name'] = [name] * len(features['distance_matrices'])
        features['coordination_number'] = [len(number)] * len(features['distance_matrices'])

        writePDB(os.path.join(output_dir, filename + '_core.pdb.gz'), core)
        with open(os.path.join(output_dir, filename + '_features.pkl'), 'wb') as f:
            pickle.dump(features,f)

        completed += 1

    print(f'{completed} core(s) identified and featurized for {pdb_file}')