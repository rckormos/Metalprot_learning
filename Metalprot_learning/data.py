"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for preprocessing and manipulating data for model training.
"""

#imports 
import os
import pickle
import itertools
import numpy as np
import torch

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

def process_features(path2features: str):
    """Reads in pickle files containing features for cores and writes them into model-readable form.

    Args:
        path2features (str): Directory containing feature files.

    Returns:
        X (np.ndarray): 
        Y (np.ndarray): 
    """
    X = []
    Y = []
    
    feature_files = [os.path.join(path2features, file) for file in os.listdir(path2features) if 'features.pkl' in file] #extract feature files
    counter = 0
    for file in feature_files:
        counter += 1
        with open(file, 'rb') as f:
            data = pickle.load(f)

        distance_mat = data['full'] #get distance matrices, encodings, and labels
        encoding = data['encoding'].squeeze()
        label = data['label'].squeeze()
        resnums = data['resnums'].squeeze()

        x = list(np.concatenate((distance_mat.flatten(), encoding))) #reshape and merge as necessary
        y = list(label)
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    np.savetxt(os.path.join(path2features, 'observations.txt'), X)
    np.savetxt(os.path.join(path2features, 'labels.txt'), Y)    

    return X, Y    

def split_data(X: np.ndarray, y: np.ndarray, train_prop=0.8):
    """Splits data into training and test sets.

    Args:
        X (np.ndarray): Observation data.
        y (np.ndarray): Label data.
        train_size (float, optional): The proportion of data to be paritioned into the training set. Defaults to 0.8.

    Returns:
        training_data (__main__.DistanceData): Dataset object of training data.
        validation_data (__main__.DistanceData): Dataset object of validation data.
    """

    training_size = int(train_prop * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    training_indices, val_indices = indices[:training_size], indices[training_size:]
    X_train, y_train, X_val, y_val = X[training_indices], y[training_indices], X[val_indices], y[val_indices]
    training_data, validation_data = DistanceData(X_train, y_train), DistanceData(X_val, y_val)

    return training_data, validation_data