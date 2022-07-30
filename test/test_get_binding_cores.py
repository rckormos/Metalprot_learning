#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains unit tests for functions in get_binding_cores.py
"""

from scipy.fftpack import idst
from Metalprot_learning.core_generator import *
from prody import *
import os
import numpy as np


def load_data():
    "Helper function for loading structures"
    data_path = '/Users/jonathanzhang/Documents/ucsf/degrado/data/metalprot_learning/src'
    pdbs = [os.path.join(data_path, file) for file in os.listdir(data_path) if '.pdb' in file]
    return pdbs 

<<<<<<< HEAD
def extract_cores_test(cores, names, coordination_numbers, no_neighbors, coordinating_resis):
    max_resis = coordinating_resis + (2 * no_neighbors * coordinating_resis)
=======
def extract_positive_cores_test(cores, names, no_neighbors, coordinating_resis):
    max_resis = (no_neighbors * 2 * coordinating_resis) + coordinating_resis
>>>>>>> d69c730f6ccba4dbadb6bec84bbc4d8f4ce728c7
    
    for core, name, coordination_number in zip(cores, names, coordination_numbers):
        no_resis = len(core.select('name CA'))
        assert no_resis <= max_resis 
        assert name
        assert coordination_number

def compute_labels_test(label, core, metal_name, no_neighbors, coordinating_resis, identifiers):
    max_atoms = 4*((no_neighbors * 2 * coordinating_resis) + coordinating_resis)
    metal_sel = core.select('hetero').select(f'name {metal_name}')
    tracker = 0
    tol = 10e-6
    label = label.squeeze()
    for identifier in identifiers:
        for atom in ['N', 'CA', 'C', 'O']:
            atom_sel = core.select(f'chain {identifier[1]}').select(f'resnum {identifier[0]}').select(f'name {atom}')
            distance = buildDistMatrix(metal_sel, atom_sel)[0,0]
            assert abs(distance - label[tracker]) < tol
            tracker += 1
    assert len(label) == max_atoms

def compute_distance_matrices_test(dist_mat, identifiers, core, no_neighbors, coordinating_resis):
    max_atoms = 4*((no_neighbors * 2 * coordinating_resis) + coordinating_resis)
    tol =10e-6

    row_index = 0
    column_index = 0
    for id1 in identifiers:
        for atom1 in ['N', 'CA', 'C', 'O']:
            for id2 in identifiers:
                for atom2 in ['N', 'CA', 'C', 'O']:
                    sel1 = core.select(f'chain {id1[1]}').select(f'resnum {id1[0]}').select(f'name {atom1}')
                    sel2 = core.select(f'chain {id2[1]}').select(f'resnum {id2[0]}').select(f'name {atom2}')
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

def permute_features_test(features, dist_mat, label, encoding, no_neighbors, coordinating_resis, ids):
    observations = features['full_observations']
    labels = features['full_labels']
    id_permutations = features['binding_core_identifier_permutations']

    max_resis = (no_neighbors * 2 * coordinating_resis) + coordinating_resis
    no_resis = len(id_permutations[0])
    tol = 10e-6
    label = label.squeeze()
    encoding = encoding.squeeze()

    counter = 0
    for perm_feature, perm_label, perm_ids in zip(observations, labels, id_permutations):
        counter += 1
        compressed_distance_matrix = perm_feature[0:(4*max_resis)**2]
        perm_dist_mat = np.vstack(list(np.array_split(compressed_distance_matrix, (4*max_resis))))
        perm_encoding = perm_feature[(4*max_resis)**2:].squeeze()
        perm_label = perm_label.squeeze()
        
        perm_ids = list(perm_ids)


        assert perm_dist_mat.shape[0] == dist_mat.shape[0] == max_resis*4
        assert len(perm_label) == len(label) == max_resis*4
        assert len(perm_encoding) == len(encoding) == max_resis*20
        assert len(perm_ids) == len(ids) <= max_resis

        atom_indices = np.array_split(np.linspace(0, dist_mat.shape[0]-1, dist_mat.shape[0]), max_resis)

        #check permutation of distance matrix
        for id1 in ids:
            for id2 in ids:
                perm_atoms_i = atom_indices[perm_ids.index(id1)]
                perm_atoms_j = atom_indices[perm_ids.index(id2)]
                atoms_i = atom_indices[ids.index(id1)]
                atoms_j = atom_indices[ids.index(id2)]

                for perm_ind_i, ind_i in zip(perm_atoms_i, atoms_i):
                    for perm_ind_j, ind_j in zip(perm_atoms_j, atoms_j):
                        perm_dist = perm_dist_mat[int(perm_ind_i), int(perm_ind_j)]
                        dist = dist_mat[int(ind_i), int(ind_j)]
                        assert abs(dist-perm_dist) < tol

        #check permutation of label and encoding
        encoding_split = np.split(encoding, max_resis)
        perm_encoding_split = np.split(perm_encoding, max_resis)
        for id in ids:
            ind = ids.index(id)
            perm_ind = perm_ids.index(id)
            
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
    coordinating_resis = 6
    for pdb in pdbs:
        print(pdb)
<<<<<<< HEAD
        cores, names, nums = extract_cores(pdb, no_neighbors, coordinating_resis=coordinating_resis)
        extract_cores_test(cores, names, nums, no_neighbors, coordinating_resis)
        unique_cores, unique_names, unique_nums = remove_degenerate_cores(cores, names, nums)
        assert len(unique_cores) > 0

        for core, name, num in zip(unique_cores, unique_names, unique_nums):
            label = compute_labels(core, name, no_neighbors, coordinating_resis)
            compute_labels_test(label, core, name, no_neighbors, coordinating_resis)

            full_dist_mat, binding_core_resnums = compute_distance_matrices(core, no_neighbors, coordinating_resis)
            compute_distance_matrices_test(full_dist_mat, binding_core_resnums, core, no_neighbors, coordinating_resis)
=======
        cores, names = extract_positive_cores(pdb, no_neighbors, coordinating_resis)
        extract_positive_cores_test(cores, names, no_neighbors, coordinating_resis)
        unique_cores, unique_names = remove_degenerate_cores(cores, names)

        for core, name in zip(unique_cores, unique_names):
>>>>>>> d69c730f6ccba4dbadb6bec84bbc4d8f4ce728c7

            full_dist_mat, binding_core_identifiers, label = compute_distance_matrices(core, name, no_neighbors, coordinating_resis)
            compute_distance_matrices_test(full_dist_mat, binding_core_identifiers, core, no_neighbors, coordinating_resis)
            compute_labels_test(label, core, name, no_neighbors, coordinating_resis, binding_core_identifiers)
            
            encoding = onehotencode(core, no_neighbors, coordinating_resis)
            onehotencode_test(encoding, no_neighbors, coordinating_resis, core)

            features = permute_fragments(full_dist_mat, encoding, label, binding_core_identifiers)
            permute_features_test(features, full_dist_mat, label, encoding, no_neighbors, coordinating_resis, binding_core_identifiers)

            all_observations = features['full_observations']
            for key in [x for x in features.keys() if type(x) == int]:
                data = features[key]
                observation = all_observations[key]
                curr_dist_mat = data['distance']
                curr_encoding = data['encoding']
                observation_construction_test(observation, curr_dist_mat, curr_encoding)

test_all()
