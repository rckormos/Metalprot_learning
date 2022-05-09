#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains unit tests for functions in get_binding_cores.py
"""

from Metalprot_learning.core_generator import *
from prody import *
import os
import numpy as np


def load_data():
    "Helper function for loading structures"
    data_path = '/Users/jonathanzhang/Documents/ucsf/degrado/data/metalprot_learning/ZN_binding_cores/src'
    pdbs = [os.path.join(data_path, file) for file in os.listdir(data_path) if '.pdb' in file]
    return pdbs 

def extract_positive_cores_test(cores, names, no_neighbors, coordinating_resis):
    max_resis = (no_neighbors * 2 * coordinating_resis) + coordinating_resis
    
    for core, name in zip(cores, names):
        no_resis = len(core.select('name CA'))
        assert no_resis <= max_resis 
        assert name

def compute_labels_test(label, core, metal_name, no_neighbors, coordinating_resis):
    max_atoms = 4*((no_neighbors * 2 * coordinating_resis) + coordinating_resis)
    resnums = core.select('name CA').getResindices()
    metal_sel = core.select('hetero').select(f'name {metal_name}')
    tracker = 0
    tol = 10e-6
    label = label.squeeze()
    for resnum in resnums:
        for atom in ['N', 'CA', 'C', 'O']:
            atom_sel = core.select(f'resindex {resnum}').select(f'name {atom}')
            distance = buildDistMatrix(metal_sel, atom_sel)[0,0]
            assert abs(distance - label[tracker]) < tol
            tracker += 1

    assert len(label) == max_atoms

def compute_distance_matrices_test(dist_mat, resnums, core, no_neighbors, coordinating_resis):
    max_atoms = 4*((no_neighbors * 2 * coordinating_resis) + coordinating_resis)
    tol =10e-6

    row_index = 0
    column_index = 0
    for resnum1 in resnums:
        for atom1 in ['N', 'CA', 'C', 'O']:
            for resnum2 in resnums:
                for atom2 in ['N', 'CA', 'C', 'O']:
                    sel1 = core.select(f'resindex {resnum1}').select(f'name {atom1}')
                    sel2 = core.select(f'resindex {resnum2}').select(f'name {atom2}')
                    distance = buildDistMatrix(sel1, sel2)[0,0]
                    assert abs(distance - dist_mat[row_index, column_index]) < tol
                    assert abs(distance - dist_mat[column_index, row_index]) < tol
                    column_index += 1
            row_index += 1
            column_index = 0 
    
    assert dist_mat.shape[1] == dist_mat.shape[0] == max_atoms

def onehotencode_test(encoding, no_neighbors, coordinating_resis, core):
    threelettercodes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', \
                        'MET','PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

    resnums = core.select('name CA').getResnums()
    seq = core.select('name CA').getResnames()
    max_resis = (no_neighbors * 2 * coordinating_resis) + coordinating_resis
    no_resis = len(resnums)
    encoding = encoding.squeeze()
    assert len(encoding) == 20*max_resis

    splits = np.array_split(encoding, max_resis)

    if no_resis < max_resis:
        for empty in splits[no_resis:]:
            assert set(list(empty)) == {0}

    for split, res in zip(splits[0:no_resis], seq):
        index = np.where(split == 1)[0][0]
        assert res == threelettercodes[index]

def permute_features_test(features, dist_mat, label, encoding, no_neighbors, coordinating_resis, resnums):
    keys = [key for key in features.keys() if type(key) == int]
    max_resis = (no_neighbors * 2 * coordinating_resis) + coordinating_resis
    no_resis = len(resnums)
    tol = 10e-6

    label = label.squeeze()
    encoding = encoding.squeeze()

    counter = 0
    for key in keys:
        counter += 1
        permuted_feature = features[key]
        perm_dist_mat = permuted_feature['distance']
        assert perm_dist_mat.shape[0] == dist_mat.shape[0] == max_resis*4

        perm_label = permuted_feature['label'].squeeze()
        assert len(perm_label) == len(label) == max_resis*4

        perm_encoding = permuted_feature['encoding'].squeeze()
        assert len(perm_encoding) == len(encoding) == max_resis*20

        perm_resnums = permuted_feature['resindices']
        assert len(perm_resnums) == len(resnums) <= max_resis

        atom_indices = np.array_split(np.linspace(0, dist_mat.shape[0]-1, dist_mat.shape[0]), max_resis)

        #check permutation of distance matrix
        for resnum1 in resnums:
            for resnum2 in resnums:
                perm_atoms_i = atom_indices[perm_resnums.index(resnum1)]
                perm_atoms_j = atom_indices[perm_resnums.index(resnum2)]
                atoms_i = atom_indices[np.where(resnums == resnum1)[0][0]]
                atoms_j = atom_indices[np.where(resnums == resnum2)[0][0]]

                for perm_ind_i, ind_i in zip(perm_atoms_i, atoms_i):
                    for perm_ind_j, ind_j in zip(perm_atoms_j, atoms_j):
                        perm_dist = perm_dist_mat[int(perm_ind_i), int(perm_ind_j)]
                        dist = dist_mat[int(ind_i), int(ind_j)]
                        assert abs(dist-perm_dist) < tol

        
        #check permutation of label and encoding
        encoding_split = np.split(encoding, max_resis)
        perm_encoding_split = np.split(perm_encoding, max_resis)
        for resnum in resnums:
            ind = np.where(resnums == resnum)[0][0]
            perm_ind = perm_resnums.index(resnum)
            
            for i in range(0,4):
                assert abs(label[ind*4+i] - perm_label[perm_ind*4+i]) < tol
            
            res = encoding_split[ind]
            perm_res = perm_encoding_split[perm_ind]
            for i,j in zip(res, perm_res):
                assert i == j

        if no_resis < max_resis:
            for i in range(no_resis, max_resis):
                assert set(perm_encoding_split[i]) == set(encoding_split[i]) == {0}

def observation_construction_test(observation, dist_mat, encoding):

    assert len(observation.squeeze()) == len(encoding.squeeze()) + (dist_mat.shape[0] * dist_mat.shape[1])

    indexer = 0
    for i in range(0, dist_mat.shape[0]):
        for j in range(0, dist_mat.shape[1]):
            assert dist_mat[i,j] == observation.squeeze()[indexer]
            indexer += 1

    for i in range(0, len(encoding.squeeze())):
        assert encoding.squeeze()[i] == observation.squeeze()[i + (dist_mat.shape[0] * dist_mat.shape[1])]

def test_all():
    "Main function that implements all unit tests."

    pdbs = load_data()
    no_neighbors = 1
    coordinating_resis = 4
    for pdb in pdbs:
        print(pdb)
        cores, names = extract_positive_cores(pdb, no_neighbors, coordinating_resis)
        extract_positive_cores_test(cores, names, no_neighbors, coordinating_resis)
        unique_cores, unique_names = remove_degenerate_cores(cores, names)

        for core, name in zip(unique_cores, unique_names):

            full_dist_mat, binding_core_resnums, label, coords = compute_distance_matrices(core, name, no_neighbors, coordinating_resis)
            compute_distance_matrices_test(full_dist_mat, binding_core_resnums, core, no_neighbors, coordinating_resis)
            compute_labels_test(label, core, name, no_neighbors, coordinating_resis)
            
            encoding = onehotencode(core, no_neighbors, coordinating_resis)
            onehotencode_test(encoding, no_neighbors, coordinating_resis, core)

            features = permute_features(full_dist_mat, encoding, label, binding_core_resnums)
            permute_features_test(features, full_dist_mat, label, encoding, no_neighbors, coordinating_resis, binding_core_resnums)

            all_observations = features['full_observations']
            for key in [x for x in features.keys() if type(x) == int]:
                data = features[key]
                observation = all_observations[key]
                curr_dist_mat = data['distance']
                curr_encoding = data['encoding']
                observation_construction_test(observation, curr_dist_mat, curr_encoding)

test_all()
