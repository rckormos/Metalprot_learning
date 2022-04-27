#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Compiles training examples into an observation and label matrix.
"""

#imports
import numpy as np
import pickle
import os

def sample(core_observations: np.ndarray, core_labels: np.ndarray, max_permutations, seed):
    """Addresses oversampling of cores with large permutation numbers.

    Args:
        core_observations (np.ndarray): Observation matrix for a given core. Rows are indexed by observation, columns are indexed by feature.
        core_labels (np.ndarray): Label matrix for a given core. 

    Returns:
        weighted_observations (np.ndarray): max_permutations x m observation matrix, where m is number of features. 
        weighted_labels (np.ndarray): Label matrix with max_permutations rows.
    """

    if core_observations.shape[0] == max_permutations:
        weighted_observations = core_observations
        weighted_labels = core_labels

    else:
        rows = np.linspace(0, core_observations.shape[0] - 1, core_observations.shape[0])

        np.random.seed(seed)
        sampled_rows = np.random.choice(rows, max_permutations - core_observations.shape[0], replace=True) #sample, with replacement, rows from observation matrix
        
        weighted_observations = core_observations
        weighted_labels = core_labels
        for row_index in sampled_rows:
            weighted_observations = np.vstack([weighted_observations, core_observations[int(row_index)]])
            weighted_labels = np.vstack([weighted_labels, core_labels[int(row_index)]])

        assert weighted_labels.shape[0] == weighted_observations.shape[0] == max_permutations

    return weighted_observations, weighted_labels

def compile_data(path2features: str, max_permutations=24, seed=42):
    """Reads in pickle files containing features for cores and writes them into model-readable form.

    Args:
        path2features (str): Directory containing feature files.
        max_permutations (int, optional): Upper limit on maximum permutation number. In our initial formulation, we are addressing 2-4 coordinate binding sites. Therefore, since 4! = 24, there are a maximum of 24 possible permutations for a 4-coordinate binding site. Defaults to 24.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        X (np.ndarray): Matrix with rows indexed by observations and columns indexed by features.
        Y (np.ndarray): Matrix with rows indexed by observations and columns indexed by labels.
    """
    
    failed = []
    feature_files = [os.path.join(path2features, file) for file in os.listdir(path2features) if 'features.pkl' in file] #extract feature files
    for iteration, file in enumerate(feature_files):
        print(file, iteration)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        if iteration == 0:
            X_unweighted = data['full_observations']
            Y_unweighted = data['full_labels']
            X, Y = sample(X_unweighted, Y_unweighted, max_permutations, seed)

        else:
            x_unweighted = data['full_observations']
            y_unweighted = data['full_labels']

            try:
                x_weighted, y_weighted = sample(x_unweighted, y_unweighted, max_permutations, seed)
                X = np.vstack([X, x_weighted])
                Y = np.vstack([Y, y_weighted])

            except:
                failed.append(file)

    np.save(os.path.join(path2features, 'observations'), X)
    np.save(os.path.join(path2features, 'labels'), Y)    
    
    if len(failed) > 0:
        with open(os.path.join(path2features, 'failed_process.txt'), 'w') as f:
            f.write('\n'.join(failed))

if __name__ == '__main__':
    path2features = '/wynton/home/rotation/jzhang1198/data/ZN_binding_cores'
    compile_data(path2features)