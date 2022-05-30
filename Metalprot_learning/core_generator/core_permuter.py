"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for permuting distance matrices, labels, and encodings for metal binding cores of positive examples.
"""

import numpy as np
from prody import *
import itertools

def _identify_fragments(binding_core_identifiers: list):
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

def _permute_dist_mat(dist_mat: np.ndarray, atom_index_permutation: list):
    permuted_dist_mat = np.zeros(dist_mat.shape) 
    for i, atom_indi in enumerate(atom_index_permutation):
        for j, atom_indj in enumerate(atom_index_permutation):
            permuted_dist_mat[i,j] = dist_mat[int(atom_indi), int(atom_indj)]
    
    return permuted_dist_mat

def _permute_encoding(encoding: np.ndarray, fragment_indices, permutation, resnums):
    split_encoding = np.array_split(encoding.squeeze(), encoding.shape[1]/20)
    _permuted_encoding = sum(sum([[list(split_encoding[i]) for i in fragment_indices[j]] for j in permutation], []), []) #permute the encoding by fragment
    zeros = np.zeros(20 * (len(split_encoding) - len(resnums[~np.isnan(resnums)])))
    permuted_encoding = np.concatenate((_permuted_encoding, zeros)) #pad encoding with zeroes to standardize shape
    assert len(permuted_encoding) == len(encoding.squeeze())
    
    return permuted_encoding

def _permute_label(label, permutation, atom_indices: list, fragment_indices):

    permuted_label = np.array([])
    for i in permutation:
        frag = fragment_indices[i]
        for j in frag:
            atoms = atom_indices[j]
            for atom in atoms:
                permuted_label = np.append(permuted_label, label[int(atom)])

    return permuted_label

def permute_fragments(dist_mat: np.ndarray, noised_dist_mat: np.ndarray, encoding: np.ndarray, label: np.ndarray, noised_label: np.ndarray, resnums: np.ndarray, chids: np.ndarray, file: str, coord: np.ndarray):
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
    observations, noised_observations, labels, noised_labels, resnum_perms, chid_perms, sources, metal_coords = [], [], [], [], [], [], [], []

    end = len(resnums[~np.isnan(resnums)])
    binding_core_identifiers = [(resnum, chid) for resnum, chid in zip(resnums[0:end], chids[0:end])]

    fragment_indices = _identify_fragments(binding_core_identifiers)
    fragment_permutations = itertools.permutations(list(range(0,len(fragment_indices)))) #get permutations of fragment indices
    atom_indices = np.split(np.linspace(0, len(binding_core_identifiers)*4-1, len(binding_core_identifiers)*4), len(binding_core_identifiers)) #for each residue in the binding core, get indices of backbone atoms
    label = label.squeeze()
    noised_label = noised_label.squeeze()
    for index, permutation in enumerate(fragment_permutations):

        fragment_index_permutation = sum([fragment_indices[i] for i in permutation], []) #get the fragment permutation defined by fragment_index_permutation
        atom_index_permutation = sum([list(atom_indices[i]) for i in fragment_index_permutation], []) 

        observations.append(np.concatenate((_permute_dist_mat(dist_mat, atom_index_permutation).flatten(), _permute_encoding(encoding, fragment_indices, permutation, resnums).squeeze())))
        noised_observations.append(np.concatenate((_permute_dist_mat(noised_dist_mat, atom_index_permutation).flatten(), _permute_encoding(encoding, fragment_indices, permutation, resnums).squeeze())))
        labels.append(_permute_label(label, permutation, atom_indices, fragment_indices).squeeze())
        noised_labels.append(_permute_label(noised_label, permutation, atom_indices, fragment_indices).squeeze())

        identifier_permutations = [binding_core_identifiers[i] for i in fragment_index_permutation]
        resnum_perms.append(np.append(np.array([tup[0] for tup in identifier_permutations]), np.array([np.nan]*(len(resnums)-len(identifier_permutations)))))
        chid_perms.append(np.append(np.array([tup[1] for tup in identifier_permutations]), np.array(['']*(len(chids)-len(identifier_permutations)))))
        sources.append(file)
        metal_coords.append(coord)

    return observations, labels, noised_observations, noised_labels, resnum_perms, chid_perms, sources, metal_coords