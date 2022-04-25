#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Compiles training examples into an observation and label matrix.
"""

#imports
import numpy as np
import pickle
import os

def compile_data(path2features: str):
    """Reads in pickle files containing features for cores and writes them into model-readable form.

    Args:
        path2features (str): Directory containing feature files.

    Returns:
        X (np.ndarray): Matrix with rows indexed by observations and columns indexed by features.
        Y (np.ndarray): Matrix with rows indexed by observations and columns indexed by labels.
    """
    X = []
    Y = []
    
    feature_files = [os.path.join(path2features, file) for file in os.listdir(path2features) if 'features.pkl' in file] #extract feature files
    for iteration, file in enumerate(feature_files):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        if iteration == 0:
            X = data['full_observations']
            Y = data['full_labels']

        else:
            X = np.vstack([X, data['full_observations']])
            Y = np.vstack([Y, data['full_labels']])

    np.save(os.path.join(path2features, 'observations'), X)
    np.save(os.path.join(path2features, 'labels'), Y)    

if __name__ == '__main__':
    path2features = ''
    compile_data(path2features)