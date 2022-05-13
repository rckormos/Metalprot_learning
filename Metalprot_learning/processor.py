"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for processing data into a model-readable form.
"""

#imports 
import numpy as np
import pickle
import os

def sample(core_observations: np.ndarray, core_labels: np.ndarray, binding_core_identifier_permutations: list, max_permutations, seed):
    """Addresses oversampling of cores with large permutation numbers.

    Args:
        core_observations (np.ndarray): Observation matrix for a given core. Rows are indexed by observation, columns are indexed by feature.
        core_labels (np.ndarray): Label matrix for a given core. 
        resindex_permutations (np.ndarray): Matrix with rows indexed by resindex permutation for a given observation.
        resnum_permutations (np.ndarray): Matrix with rows indexed by resnum permutation for a given observation.

    Returns:
        weighted_observations (np.ndarray): max_permutations x m observation matrix, where m is number of features. 
        weighted_labels (np.ndarray): Label matrix with max_permutations rows.
        weighted_permutations (list): List of numpy arrays containing resindex permutations for a given observation/label pair.
    """

    weighted_observations = core_observations
    weighted_labels = core_labels
    weighted_identifier_perms = binding_core_identifier_permutations

    if core_observations.shape[0] == max_permutations:
        pass

    else:
        rows = np.linspace(0, core_observations.shape[0] - 1, core_observations.shape[0])

        np.random.seed(seed)
        sampled_rows = np.random.choice(rows, max_permutations - core_observations.shape[0], replace=True) #sample, with replacement, rows from observation matrix
        for row_index in sampled_rows:
            weighted_observations, weighted_labels = np.vstack([weighted_observations, core_observations[int(row_index)]]), np.vstack([weighted_labels, core_labels[int(row_index)]])
            weighted_identifier_perms.append(binding_core_identifier_permutations[int(row_index)])

        assert weighted_labels.shape[0] == weighted_observations.shape[0] == max_permutations == len(weighted_identifier_perms)

    return weighted_observations, weighted_labels, weighted_identifier_perms

def compile_data(path2features: str, job_id: int, feature_files, max_permutations=24, seed=42):
    """Reads in pickle files containing features for cores and writes them to a pickle file containing model-readable features with an index.

    Args:
        path2features (str): Directory containing feature files.
        max_permutations (int, optional): Upper limit on maximum permutation number. In our initial formulation, we are addressing 2-4 coordinate binding sites. Therefore, since 4! = 24, there are a maximum of 24 possible permutations for a 4-coordinate binding site. Defaults to 24.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    
    failed = []
    compiled_features = {}
    for file in feature_files:
        print(file)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        X_unweighted = data['full_observations']
        Y_unweighted = data['full_labels']
        binding_core_identifier_permutations_unweighted = data['binding_core_identifier_permutations']

        try:
            x_weighted, y_weighted, binding_core_identifier_permutations_weighted = sample(X_unweighted, Y_unweighted, binding_core_identifier_permutations_unweighted, max_permutations, seed)

            if 'X' not in locals():
                X = x_weighted
                Y = y_weighted
                binding_core_identifier_permutations = binding_core_identifier_permutations_weighted
                pointers = [data['source']] * len(X)
                assert len(pointers) == len(binding_core_identifier_permutations) == len(X) == len(Y)

            else:
                X = np.vstack([X, x_weighted])
                Y = np.vstack([Y, y_weighted])
                binding_core_identifier_permutations += binding_core_identifier_permutations_weighted
                pointers += [data['source']] * len(x_weighted)
                assert len(pointers) == len(binding_core_identifier_permutations) == len(X) == len(Y)

        except:
            failed.append(file)

    compiled_features['pointers'] = pointers
    compiled_features['binding_core_identifiers'] = binding_core_identifier_permutations
    compiled_features['observations'] = X
    compiled_features['labels'] = Y

    with open(os.path.join(path2features, f'compiled_features{job_id}.pkl'), 'wb') as f:
        pickle.dump(compiled_features, f)
    
    if len(failed) > 0:
        with open(os.path.join(path2features, f'failed_process.txt'), 'a') as f:
            f.write('\n'.join(failed) + '\n')