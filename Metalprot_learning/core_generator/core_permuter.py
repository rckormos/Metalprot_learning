"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for permuting distance matrices, labels, and encodings for metal binding cores of positive examples.
"""

import numpy as np
from prody import *
import itertools

def get_contiguous_resnums(resnums: np.ndarray):
    """Helper function for permute_features. 

    Args:
        resnums (np.ndarray): Array of resnums in the order they appear when calling core.getResnums().

    Returns:
        fragment (list): List of sorted lists containing indices of contiguous stretches of resnums. 
    """

    resnums = list(resnums)
    temp = resnums[:]
    fragment_indices = []
    while len(temp) != 0:
        for i in range(0,len(temp)):
            if i == 0:
                fragment = [temp[i]]

            elif 1 in set([abs(temp[i] - j) for j in fragment]):
                fragment.append(temp[i])

        fragment = list(set(fragment))
        fragment.sort()
        one_fragment_indices = [resnums.index(i) for i in fragment]
        fragment_indices.append(one_fragment_indices)

        for index in fragment:
            temp.remove(index)
    
    return fragment_indices

def permute_features(dist_mat: np.ndarray, encoding: np.ndarray, label: np.ndarray, resnums: np.ndarray):
    """Computes fragment permutations for input features and labels. 

    Args:
        dist_mat (np.ndarray): Distance matrix.
        encoding (np.ndarray): One-hot encoding of sequence.
        label (np.ndarray): Backbone atom distances to the metal.
        resnums (np.ndarray): Resnums of core atoms in the order the appear when calling core.getResnums().

    Returns:
        all_features (dict): Dictionary containing compiled observation and label matrices for a training example as well as distance matrices and labels for individual permutations.
    """
    all_features = {}
    full_observations = []
    full_labels = []

    fragment_indices = get_contiguous_resnums(resnums)
    fragment_index_permutations = itertools.permutations(list(range(0,len(fragment_indices))))
    atom_indices = np.split(np.linspace(0, len(resnums)*4-1, len(resnums)*4), len(resnums))
    label = label.squeeze()
    for index, index_permutation in enumerate(fragment_index_permutations):
        feature = {}
        permutation = sum([fragment_indices[i] for i in index_permutation], [])
        atom_ind_permutation = sum([list(atom_indices[i]) for i in permutation], [])
        permuted_dist_mat = np.zeros(dist_mat.shape)

        for i, atom_indi in enumerate(atom_ind_permutation):
            for j, atom_indj in enumerate(atom_ind_permutation):
                permuted_dist_mat[i,j] = dist_mat[int(atom_indi), int(atom_indj)]

        if index == 0:
            for i in range(0,permuted_dist_mat.shape[0]):
                for j in range(0, permuted_dist_mat.shape[1]):
                    assert permuted_dist_mat[i,j] == dist_mat[i,j]

        split_encoding = np.array_split(encoding.squeeze(), encoding.shape[1]/20)
        _permuted_encoding = sum(sum([[list(split_encoding[i]) for i in fragment_indices[j]] for j in index_permutation], []), [])
        zeros = np.zeros(20 * (len(split_encoding) - len(resnums)))
        permuted_encoding = np.concatenate((_permuted_encoding, zeros))

        assert len(permuted_encoding) == len(encoding.squeeze())

        permuted_label = np.array([])
        for i in index_permutation:
            frag = fragment_indices[i]
            for j in frag:
                atoms = atom_indices[j]
                for atom in atoms:
                    permuted_label = np.append(permuted_label, label[int(atom)])

        permuted_label = np.append(permuted_label, np.zeros(len(label) - len(permuted_label)))

        feature['distance'] = permuted_dist_mat
        feature['encoding'] = permuted_encoding
        feature['label'] = permuted_label
        feature['resnums'] = [resnums[i] for i in permutation]

        full_observations.append(list(np.concatenate((permuted_dist_mat.flatten(), permuted_encoding))))
        full_labels.append(list(permuted_label))

        all_features[index] = feature
    all_features['full_observations'] = np.array(full_observations)
    all_features['full_labels'] = np.array(full_labels)
    return all_features