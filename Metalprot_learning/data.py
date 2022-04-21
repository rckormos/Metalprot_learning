"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for preprocessing data for model training.
"""

#imports 
import os
import pickle
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DistanceData(torch.utils.Dataset):
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

        x = list(np.concatenate((distance_mat.flatten(), encoding))) #reshape and merge as necessary
        y = list(label)
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    np.savetxt(os.path.join(path2features, 'observations'), X)
    np.savetxt(os.path.join(path2features, 'labels'), Y)    

    return X, Y    

def split_data(X: np.ndarray, y: np.ndarray, train_size=0.8):
    full = np.concatenate((X,y), axis=1)
    split_index = X.shape[1]
    train_dataset, test_dataset = torch.utils.data.random_split(full, [train_size, 1-train_size])

    X_train = train_dataset[:,0:split_index]
    y_train = train_dataset[:,split_index:]
    training_data = DistanceData(X_train, y_train)

    X_val = test_dataset[:,0:split_index]
    y_val = test_dataset[:,split_index:]
    validation_data = DistanceData(X_val, y_val)

    return training_data, validation_data