"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for processing data into a model-readable form.
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

def compile_data(path2features: str, feature_files, max_permutations=24, seed=42):
    """Reads in pickle files containing features for cores and writes them into model-readable form. Also writes an index.pkl file containing source pdb files.

    Args:
        path2features (str): Directory containing feature files.
        max_permutations (int, optional): Upper limit on maximum permutation number. In our initial formulation, we are addressing 2-4 coordinate binding sites. Therefore, since 4! = 24, there are a maximum of 24 possible permutations for a 4-coordinate binding site. Defaults to 24.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    
    failed = []
    for iteration, file in enumerate(feature_files):
        print(file, iteration)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        if iteration == 0:
            X_unweighted = data['full_observations']
            Y_unweighted = data['full_labels']
            X, Y = sample(X_unweighted, Y_unweighted, max_permutations, seed)

            pointers = [data['source']] * len(X)

            assert len(pointers) == len(X)

        else:
            x_unweighted = data['full_observations']
            y_unweighted = data['full_labels']

            try:
                x_weighted, y_weighted = sample(x_unweighted, y_unweighted, max_permutations, seed)
                X = np.vstack([X, x_weighted])
                Y = np.vstack([Y, y_weighted])

                pointers += [data['source']] * len(x_weighted)
                
                assert len(pointers) == len(x_weighted)

            except:
                failed.append(file)

    assert len(pointers) == len(X) == len(Y)

    np.save(os.path.join(path2features, 'observations'), X)
    np.save(os.path.join(path2features, 'labels'), Y)    
    with open(os.path.join(path2features, 'index.pkl'), 'wb') as f:
        pickle.dump(pointers, f)
    
    if len(failed) > 0:
        with open(os.path.join(path2features, 'failed_process.txt'), 'w') as f:
            f.write('\n'.join(failed))