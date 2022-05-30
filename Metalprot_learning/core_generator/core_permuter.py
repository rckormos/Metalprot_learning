"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for permuting distance matrices, labels, and encodings for metal binding cores of positive examples.
"""

import numpy as np
from prody import *
import itertools

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

def permute_fragments(dist_mat: np.ndarray, encoding: np.ndarray, label: np.ndarray, binding_core_identifiers: list):
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
    all_features = {}
    full_observations = []
    full_labels = []
    binding_core_identifier_permutations = []

    fragment_indices = identify_fragments(binding_core_identifiers)
    fragment_permutations = itertools.permutations(list(range(0,len(fragment_indices)))) #get permutations of fragment indices
    atom_indices = np.split(np.linspace(0, len(binding_core_identifiers)*4-1, len(binding_core_identifiers)*4), len(binding_core_identifiers)) #for each residue in the binding core, get indices of backbone atoms
    label = label.squeeze()
    for index, permutation in enumerate(fragment_permutations):
        fragment_index_permutation = sum([fragment_indices[i] for i in permutation], []) #get the fragment permutation defined by fragment_index_permutation
        atom_index_permutation = sum([list(atom_indices[i]) for i in fragment_index_permutation], []) 

        permuted_dist_mat = np.zeros(dist_mat.shape) 
        for i, atom_indi in enumerate(atom_index_permutation):
            for j, atom_indj in enumerate(atom_index_permutation):
                permuted_dist_mat[i,j] = dist_mat[int(atom_indi), int(atom_indj)]

        split_encoding = np.array_split(encoding.squeeze(), encoding.shape[1]/20)
        _permuted_encoding = sum(sum([[list(split_encoding[i]) for i in fragment_indices[j]] for j in permutation], []), []) #permute the encoding by fragment
        zeros = np.zeros(20 * (len(split_encoding) - len(binding_core_identifiers)))
        permuted_encoding = np.concatenate((_permuted_encoding, zeros)) #pad encoding with zeroes to standardize shape
        assert len(permuted_encoding) == len(encoding.squeeze())

        permuted_label = np.array([])
        for i in permutation:
            frag = fragment_indices[i]
            for j in frag:
                atoms = atom_indices[j]
                for atom in atoms:
                    permuted_label = np.append(permuted_label, label[int(atom)])

        permuted_label = np.append(permuted_label, np.zeros(len(label) - len(permuted_label)))
        binding_core_identifier_permutations.append([binding_core_identifiers[i] for i in fragment_index_permutation])
        full_observations.append(np.concatenate((permuted_dist_mat.flatten(), permuted_encoding)))
        full_labels.append(permuted_label)

    all_features['observations'] = full_observations
    all_features['labels'] = full_labels
    all_features['identifiers'] = binding_core_identifier_permutations
    return all_features