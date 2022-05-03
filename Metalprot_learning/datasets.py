"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading and splitting data for model training and validation.
"""

#imports
import torch
import numpy as np
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

def split_data(observation_file: str, label_file: str, index_file: str, partitions: tuple, seed: int):
    """Splits data into training and test sets.

    Args:
        observation_file (str): Path to observations.npy file.
        label_file (str): Path to labels.npy file.
        index_file (str): Path to index.pkl file.
        partitions (tuple): Tuple containing proportion to partition into training, testing, and validation sets respectively.
        seed (int): The random seed for splitting.

    Returns:
        training_data (tuple): Tuple containing training observations and labels.
        testing_data (tuple): Tuple containing testing data and labels.
        validation_data (tuple): Tuple containing validation data and labels.
        train_index (dict): Dictionary containing PDB IDs, metal names, and metal coordinates indexed by training observations.
        test_index (dict): Dictionary containing PDB IDs, metal names, and metal coordinates indexed by test observations.
        val_index (dict): Dictionary containing PDB IDs, metal names, and metal coordinates indexed by validation observations.
    """

    #load data
    X = np.load(observation_file)
    y = np.load(label_file)
    with open(index_file, 'rb') as f:
        index = pickle.load(f)

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
    
    train_index = {}
    test_index = {}
    val_index = {}
    for key in index.keys():
        train_index[key] = index[key][training_indices]
        test_index[key] = index[key][test_indices]
        val_index[key] = index[key][val_indices]

    assert sum([i.shape[0] for i in [X_train, X_test, X_val]]) == sum([i.shape[0] for i in [y_train, y_test, y_val]]) == X.shape[0]

    training_data, testing_data, validation_data = (X_train, y_train), (X_test, y_test), (X_val, y_val)

    return training_data, testing_data, validation_data, train_index, test_index, val_index