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

def get_contiguous_resnums(resnums: np.ndarray):
    resnums = list(resnums)
    temp = resnums[:]
    fragments = []
    for resnum in temp:
        fragment = []
        temp.remove(resnum)
        fragment.append(resnum)
        queue = [i for i in temp if abs(i-resnum)==1]

        while len(queue) != 0:
            current = queue.pop()
            fragment.append(current)
            temp.remove(current)
            queue += [i for i in temp if abs(i-current)==1]

        fragment.sort()
        fragments.append(fragment)

    fragment_indices = []
    for fragment in fragments:
        fragment_indices.append([resnums.index(i) for i in fragment])
    
    return fragment_indices

def permute_features(dist_mat: np.ndarray, encoding: np.ndarray, label: np.ndarray, resnums: np.ndarray):
    all_features = {}

    fragment_indices = get_contiguous_resnums(resnums)
    fragment_index_permutations = itertools.permutations(list(range(0,len(fragment_indices))))
    for index, index_permutation in enumerate(fragment_index_permutations):
        feature = {}
        permutation = sum([fragment_indices[i] for i in index_permutation], [])
        permuted_dist_mat = np.zeros(dist_mat.shape)

        r = 1
        for i in range(0,len(permutation)):
            for j in range(1+r,len(permutation)):
                permuted_dist_mat[i,j] = dist_mat[permutation[i], permutation[j]]
                permuted_dist_mat[j,i] = permuted_dist_mat[i,j]
            r += 1

        split_encoding = np.array_split(encoding.squeeze(), encoding.shape[1]/20)
        _permuted_encoding = sum(sum([[list(split_encoding[i]) for i in fragment_indices[j]] for j in index_permutation], []), [])
        zeros = np.zeros(20 * (len(split_encoding) - len(resnums)))
        permuted_encoding = np.concatenate((_permuted_encoding, zeros))

        permuted_label = []
        for i in index_permutation:
            frag = fragment_indices[i]
            for j in range(0,len(frag)):
                ind = frag[j]
                permuted_label += list(label[4*ind:4*ind+4])



        feature['distance'] = permuted_dist_mat
        feature['encoding'] = permuted_encoding
        feature['label'] = permuted_label

        all_features[index] = feature
    return all_features

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