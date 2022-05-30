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

def _split(X: np.ndarray, y: np.ndarray, permutations: list, sources: list, partitions, seed: int):

    #get all pdb ids
    ids = list(set([id.split('/')[-1].split('_')[0] for id in sources]))

    #define data partitions
    train_prop, test_prop, val_prop = partitions
    assert sum([train_prop, test_prop, val_prop]) == 1
    training_size, testing_size = int(train_prop * len(ids)), int(test_prop * len(ids))
    indices = np.random.RandomState(seed=seed).permutation(len(ids))
    training_indices, test_indices, val_indices = indices[:training_size], indices[training_size:(training_size+testing_size)], indices[(training_size + testing_size):] 

    #randomly assign cores based on pdb id
    train_ids, test_ids, val_ids = [ids[int(i)] for i in training_indices], [ids[int(i)] for i in test_indices], [ids[int(i)] for i in val_indices]
    assert set(train_ids).intersection(set(test_ids), set(val_ids)) == set(test_ids).intersection(set(train_ids), set(val_ids)) == set(val_ids).intersection(set(train_ids), set(test_ids)) == set()

    #get indices of train, test, and validation sets
    training_indices, test_indices, val_indices = np.array([index for index, element in enumerate(sources) if element.split('/')[-1].split('_')[0] in train_ids]), np.array([index for index, element in enumerate(sources) if element.split('/')[-1].split('_')[0] in test_ids]), np.array([index for index, element in enumerate(sources) if element.split('/')[-1].split('_')[0] in val_ids])
    assert set(training_indices).intersection(set(test_indices), set(val_indices)) == set(test_indices).intersection(set(training_indices), set(val_indices)) == set(val_indices).intersection(set(training_indices), set(test_indices)) == set()

    #build datasets
    X_train, y_train, X_test, y_test, X_val, y_val = X[training_indices], y[training_indices], X[test_indices], y[test_indices], X[val_indices], y[val_indices]
    train_sources, test_sources, val_sources = [sources[int(i)] for i in training_indices], [sources[int(i)] for i in test_indices], [sources[int(i)] for i in val_indices]
    train_identifier_permutations, test_identifier_permutations, val_identifier_permutations = [permutations[int(i)] for i in training_indices], [permutations[int(i)] for i in test_indices], [permutations[int(i)] for i in val_indices]
    
    assert len({X_train.shape[0], y_train.shape[0], len(train_sources), len(train_identifier_permutations)}) == 1
    assert len({X_test.shape[0], y_test.shape[0], len(test_sources), len(test_identifier_permutations)}) == 1
    assert len({X_val.shape[0], y_val.shape[0], len(val_sources), len(val_identifier_permutations)}) == 1
    assert sum([i.shape[0] for i in [X_train, X_test, X_val]]) == sum([i.shape[0] for i in [y_train, y_test, y_val]]) == X.shape[0]

    train_index = {'pointers': train_sources, 'binding_core_identifier_permutations': train_identifier_permutations}
    test_index = {'pointers': test_sources, 'binding_core_identifier_permutations': test_identifier_permutations}
    val_index = {'pointers': val_sources, 'binding_core_identifier_permutations': val_identifier_permutations}
    training_data, testing_data, validation_data = (X_train, y_train, train_index), (X_test, y_test, test_index), (X_val, y_val, val_index)

    return training_data, testing_data, validation_data, (training_indices, test_indices, val_indices) 

def split_data(features_file: str, partitions: tuple, seed: int):
    """Splits data into training and test sets.

    Args:
        features_file (str): Path to compiled_features.pkl file.
        partitions (tuple): Tuple containing proportion to partition into training, testing, and validation sets respectively.
        seed (int): The random seed for splitting.
        random (bool, optional): Determines whether the dataset is split randomly or by pdb file.

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
    binding_core_identifier_permutations = features['binding_core_identifiers']
    sources = features['pointers']

    training_data, testing_data, validation_data, (training_indices, test_indices, val_indices) = _split(X, y, binding_core_identifier_permutations, sources, partitions, seed)

    return training_data, testing_data, validation_data, (training_indices, test_indices, val_indices)