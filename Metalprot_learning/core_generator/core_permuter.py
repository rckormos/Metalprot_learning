"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for permuting distance matrices, labels, and encodings for metal binding cores of positive examples.
"""

import numpy as np
from prody import *
import itertools

def _trim(dist_mat: np.ndarray):
    trimmed = []
    for row_ind, indexer in zip(range(0, len(dist_mat)-1), range(1, len(dist_mat))):
        trimmed.append(dist_mat[row_ind][indexer:])

    trimmed = np.concatenate(trimmed)
    return trimmed

def identify_fragments(binding_core_identifiers: list):
    """Helper function for permute_features. Given a list of numbers, finds all contiguous stretches and outputs a list of lists.

    Args:
        numbers (np.ndarray): Array of resindices in the order they appear when calling core.getResindices().

    Returns:
        fragment (list): List of sorted lists containing indices of contiguous stretches of resindices. 
    """

    temp = binding_core_identifiers[:]
    fragments = []
    while len(temp) != 0:
        for i in range(0, len(temp)): #build up contiguous fragments by looking for adjacent resnums
            if i == 0:
                fragment = [temp[i]]

            elif set(temp[i][1]) == set([i[1] for i in fragment]) and 1 in set([abs(temp[i][0] - j[0]) for j in fragment]):
                fragment.append(temp[i])

        fragment = list(set(fragment)) 
        fragment.sort()
        fragment_indices = [binding_core_identifiers.index(i) for i in fragment] 
        fragments.append(fragment_indices) #build a list containing lists of indices of residues for a given fragment

        for item in fragment:
            temp.remove(item)
    
    return fragments

def _permute_matrices(dist_mat: np.ndarray, noised_dist_mat: np.ndarray, atom_ind_permutation):
    permuted_dist_mat, noised_permuted_dist_mat = np.zeros(dist_mat.shape), np.zeros(dist_mat.shape)
    for i, atom_indi in enumerate(atom_ind_permutation):
        for j, atom_indj in enumerate(atom_ind_permutation):
            permuted_dist_mat[i,j] = dist_mat[int(atom_indi), int(atom_indj)]
            noised_permuted_dist_mat[i,j] = noised_dist_mat[int(atom_indi), int(atom_indj)]

    return permuted_dist_mat, noised_permuted_dist_mat

def _permute_encodings(encoding: np.ndarray, fragment_indices, permutation, binding_core_identifiers):
    split_encoding = np.array_split(encoding.squeeze(), encoding.shape[1]/20)
    _permuted_encoding = sum(sum([[list(split_encoding[i]) for i in fragment_indices[j]] for j in permutation], []), []) #permute the encoding by fragment
    zeros = np.zeros(20 * (len(split_encoding) - len(binding_core_identifiers)))
    permuted_encoding = np.concatenate((_permuted_encoding, zeros)) #pad encoding with zeroes to standardize shape
    assert len(permuted_encoding) == len(encoding.squeeze())
    
    return permuted_encoding

def _permute_labels(label: np.ndarray, noised_label: np.ndarray, permutation, fragment_indices, atom_indices):
    permuted_label, permuted_noised_label = np.array([]), np.array([])
    for i in permutation:
        frag = fragment_indices[i]
        for j in frag:
            atoms = atom_indices[j]
            for atom in atoms:
                permuted_label = np.append(permuted_label, label[int(atom)])
                permuted_noised_label = np.append(permuted_noised_label, noised_label[int(atom)])

    permuted_label = np.append(permuted_label, np.zeros(len(label) - len(permuted_label)))
    permuted_noised_label = np.append(permuted_noised_label, np.zeros(len(label) - len(permuted_noised_label)))
    return permuted_label, permuted_noised_label

def permute_fragments(dist_mat: np.ndarray, label: np.ndarray, noised_dist_mat: np.ndarray, noised_label: np.ndarray, encoding: np.ndarray, binding_core_identifiers: list, coordinate_label: np.ndarray, c_beta: bool, trim: bool):
    """Computes fragment permutations for input features and labels. 

    Args:
        dist_mat (np.ndarray): Distance matrix.
        encoding (np.ndarray): One-hot encoding of sequence.
        label (np.ndarray): Backbone atom distances to the metal.
        resindices (np.ndarray): Resindices of core atoms in the order the appear when calling core.getResindices().
        resnums (np.ndarray): Resnums of core atoms in the order the appear when calling core.getResnums().

    Returns:
        all_features (dict): Dictionary containing compiled observation and label matrices for a training example as well as a list of permutations indexed by observation matrix row.
    """
    distance_matrices, full_labels, noised_distance_matrices, noised_labels, encodings, binding_core_identifier_permutations, coordinate_labels = [], [], [], [], [], [], []

    fragment_indices = identify_fragments(binding_core_identifiers)
    fragment_permutations = itertools.permutations(list(range(0,len(fragment_indices)))) #get permutations of fragment indices
    backbone_atoms = 5 if c_beta else 4
    atom_indices = np.split(np.linspace(0, len(binding_core_identifiers)*backbone_atoms-1, len(binding_core_identifiers)*backbone_atoms), len(binding_core_identifiers)) #for each residue in the binding core, get indices of backbone atoms
    label = label.squeeze()
    noised_label = noised_label.squeeze()
    for index, permutation in enumerate(fragment_permutations):
        fragment_index_permutation = sum([fragment_indices[i] for i in permutation], []) #get the fragment permutation defined by fragment_index_permutation
        atom_index_permutation = sum([list(atom_indices[i]) for i in fragment_index_permutation], []) 

        _permuted_dist_mat, _permuted_noised_dist_mat = _permute_matrices(dist_mat, noised_dist_mat, atom_index_permutation)
        permuted_encoding = _permute_encodings(encoding, fragment_indices, permutation, binding_core_identifiers)
        permuted_label, permuted_noised_label = _permute_labels(label, noised_label, permutation, fragment_indices, atom_indices)
        permuted_dist_mat, permuted_noised_dist_mat = (_trim(_permuted_dist_mat), _trim(_permuted_noised_dist_mat)) if trim else (_permuted_dist_mat.flatten().squeeze(), _permuted_noised_dist_mat.flatten().squeeze())

        binding_core_identifier_permutations.append([binding_core_identifiers[i] for i in fragment_index_permutation])
        coordinate_labels.append(np.array([coordinate_label[i] for i in fragment_index_permutation]))
        distance_matrices.append(permuted_dist_mat)
        noised_distance_matrices.append(permuted_noised_dist_mat)
        encodings.append(permuted_encoding.squeeze())
        full_labels.append(permuted_label)
        noised_labels.append(permuted_noised_label)

    all_features = {'distance_matrices': distance_matrices, 'noised_distance_matrices': noised_distance_matrices, 'labels': full_labels, 
                    'noised_labels': noised_labels, 'encodings': encodings, 'identifiers': binding_core_identifier_permutations, 'coordinate_labels': coordinate_labels}

    return all_features