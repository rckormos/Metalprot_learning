"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading and splitting data for model training and validation.
"""

#imports
import torch
import numpy as np

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

def split_data(observation_file: str, label_file: str, partitions: tuple, seed: int):
    """Splits data into training and test sets.

    Args:
        observation_file (str): Path to observations.npy file.
        label_file (str): Path to labels.npy file.
        partitions (tuple): Tuple containing proportion to partition into training, testing, and validation sets respectively.
        seed (int): The random seed for splitting.

    Returns:
        training_data (__main__.DistanceData): Dataset object of training data.
        testing_data (__main__.DistanceData): Dataset object of testing data.
        validation_data (__main__.DistanceData): Dataset object of validation data.
    """

    #load data
    X = np.load(observation_file)
    y = np.load(label_file)

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
    assert sum([i.shape[0] for i in [X_train, X_test, X_val]]) == sum([i.shape[0] for i in [y_train, y_test, y_val]]) == X.shape[0]

    training_data, testing_data, validation_data = DistanceData(X_train, y_train), DistanceData(X_test, y_test), DistanceData(X_val, y_val)

    return training_data, testing_data, validation_data