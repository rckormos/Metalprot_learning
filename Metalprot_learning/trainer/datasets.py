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
    permutations = features['permutations']
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
    
    train_sources, test_sources, val_sources = sources[training_indices], sources[test_indices], sources[val_indices]
    train_permutations, test_permutations, val_permutations = permutations[training_indices], permutations[test_indices], permutations[val_indices]

    assert sum([i.shape[0] for i in [X_train, X_test, X_val]]) == sum([i.shape[0] for i in [y_train, y_test, y_val]]) == X.shape[0]

    train_index = {'sources': train_sources, 'permutations': train_permutations}
    test_index = {'sources': test_sources, 'permutations': test_permutations}
    val_index = {'sources': val_sources, 'permutations': val_permutations}
    training_data, testing_data, validation_data = (X_train, y_train, train_index), (X_test, y_test, test_index), (X_val, y_val, val_index)

    return training_data, testing_data, validation_data