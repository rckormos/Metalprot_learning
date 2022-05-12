"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading and splitting data for model training and validation.
"""

#imports
import numpy as np
import torch
import pickle

class DistanceData(torch.utils.data.Dataset):
    "Custom dataset class"

    def __init__(self, observations, labels):
        self.labels = labels
        self.observations = observations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        observation = self.observations[index]
        label = self.labels[index]
        return observation, label 

def split_data(features_file: str, partitions: tuple, seed: int):
    """Splits data into training and test sets.

    Args:
        features_file (str): Path to compiled_features.pkl file.
        partitions (tuple): Tuple containing proportion to partition into training, testing, and validation sets respectively.
        seed (int): The random seed for splitting.

    Returns:
        training_data (tuple): Tuple containing numpy arrays of training observations and labels as well as indices.
        testing_data (tuple): Tuple containing numpy arrays testing data and labels as well as indices.
        validation_data (tuple): Tuple containing numpy arrays validation data and labels as well as indices.
    """

    #load data
    with open(features_file, 'rb') as f:
        features = pickle.load(f)

    X = features['observations']
    y = features['labels']
    resind_permutations = features['resindex_permutations']
    resnum_permutations = features['resnum_permutations']
    metal_coords = features['metal_coords']
    sources = features['pointers']

    #define data partitions
    train_prop, test_prop, val_prop = partitions
    assert sum([train_prop, test_prop, val_prop]) == 1
    training_size = int(train_prop * X.shape[0])
    testing_size = int(test_prop * X.shape[0])
    validation_size = int(val_prop * X.shape[0])

    #randomly assign examples to training sets
    indices = np.random.RandomState(seed=seed).permutation(X.shape[0])
    training_indices, test_indices, val_indices = indices[:training_size], indices[training_size:(training_size+testing_size)], indices[(training_size + testing_size):] 
    X_train, y_train, X_test, y_test, X_val, y_val = X[training_indices], y[training_indices], X[test_indices], y[test_indices], X[val_indices], y[val_indices]
    
    train_sources, test_sources, val_sources = [sources[int(i)] for i in training_indices], [sources[int(i)] for i in test_indices], [sources[int(i)] for i in val_indices]
    train_metal_coords, test_metal_coords, val_metal_coords = [metal_coords[int(i)] for i in training_indices], [sources[int(i)] for i in test_indices], [sources[int(i)] for i in val_indices]
    train_resind_permutations, test_resind_permutations, val_resind_permutations = [resind_permutations[int(i)] for i in training_indices], [resind_permutations[int(i)] for i in test_indices], [resind_permutations[int(i)] for i in val_indices]
    train_resnum_permutations, test_resnum_permutations, val_resnum_permutations = [resnum_permutations[int(i)] for i in training_indices], [resnum_permutations[int(i)] for i in test_indices], [resnum_permutations[int(i)] for i in val_indices]
    
    assert len({X_train.shape[0], y_train.shape[0], len(train_sources), len(train_metal_coords), len(train_resind_permutations), len(train_resnum_permutations)}) == 1
    assert len({X_test.shape[0], y_test.shape[0], len(test_sources), len(test_metal_coords), len(test_resind_permutations), len(test_resnum_permutations)}) == 1
    assert len({X_val.shape[0], y_val.shape[0], len(val_sources), len(val_metal_coords), len(val_resind_permutations), len(val_resnum_permutations)}) == 1
    assert sum([i.shape[0] for i in [X_train, X_test, X_val]]) == sum([i.shape[0] for i in [y_train, y_test, y_val]]) == X.shape[0]

    train_index = {'pointers': train_sources, 'resindex_permutations': train_resind_permutations, 'resnum_permutations': train_resnum_permutations, 'metal_coords': train_metal_coords}
    test_index = {'pointers': test_sources, 'resindex_permutations': test_resind_permutations, 'resnum_permutations': test_resnum_permutations, 'metal_coords': test_metal_coords}
    val_index = {'pointers': val_sources, 'resindex_permutations': val_resind_permutations, 'resnum_permutations': val_resnum_permutations, 'metal_coords': val_metal_coords}
    training_data, testing_data, validation_data = (X_train, y_train, train_index), (X_test, y_test, test_index), (X_val, y_val, val_index)

    return training_data, testing_data, validation_data