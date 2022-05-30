"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading and splitting data for model training and validation.
"""

#imports
import pandas as pd
import numpy as np
import torch
import pickle

class DistanceData(torch.utils.data.Dataset):
    "Custom dataset class"

    def __init__(self, set: pd.DataFrame, encodings: bool):

        if encodings:
            self.observations = np.hstack([np.vstack([array for array in set['distance_matrices']]), np.vstack([array for array in set['encodings']])])

        else:
            self.observations = np.vstack([array for array in set['distance_matrices']])

        self.labels = np.vstack([array for array in set['labels']])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        observation = self.observations[index]
        label = self.labels[index]
        return observation, label 

def sample_by_pdb(d: dict, partitions: tuple, seed: int):
    ids = list(d.keys())

    train_prop, test_prop, val_prop = partitions
    assert sum([train_prop, test_prop, val_prop]) == 1
    train_size, testing_size = int(train_prop * len(ids)), int(test_prop * len(ids))
    indices = np.random.RandomState(seed=seed).permutation(len(ids))
    train_indices, test_indices, val_indices = indices[:train_size], indices[train_size:(train_size+testing_size)], indices[(train_size + testing_size):]

    train_set, test_set, val_set = pd.concat([d[ids[ind]] for ind in train_indices]), pd.concat([d[ids[ind]] for ind in test_indices]), pd.concat([d[ids[ind]] for ind in val_indices])

    return train_set, test_set, val_set

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

    d = {}
    ids = set([i.split('/')[-1].split('_')[0] for i in features['source']])
    for id in ids:
        d[id] = features[features['source'].str.contains(id)]
   
    train_set, test_set, val_set = sample_by_pdb(d, partitions, seed)

    return train_set, test_set, val_set