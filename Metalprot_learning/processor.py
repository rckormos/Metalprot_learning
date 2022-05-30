"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for processing data into a model-readable form.
"""

#imports 
import pandas as pd
import pickle
import os

def compile_data(path2features: str, job_id: int, feature_files, permuted: bool, max_permutations=24, seed=42):
    """Reads in pickle files containing features for cores and writes them to a pickle file containing model-readable features with an index.

    Args:
        path2features (str): Directory containing feature files.
        max_permutations (int, optional): Upper limit on maximum permutation number. In our initial formulation, we are addressing 2-4 coordinate binding sites. Therefore, since 4! = 24, there are a maximum of 24 possible permutations for a 4-coordinate binding site. Defaults to 24.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    
    failed = []
    for file in feature_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)

        try:
            _features = pd.DataFrame(data)
            upsample = max_permutations - len(_features)
            _features = pd.concat([_features, _features.sample(n=upsample, replace=True, random_state=seed)]) if permuted else _features

            features = pd.concat([features, _features]) if 'features' in locals() else _features
            print(f'Successfully compiled {file}')

        except:
            failed.append(file)

    features.to_pickle(os.path.join(path2features, f'compiled_features{job_id}.pkl'))
    if len(failed) > 0:
        with open(os.path.join(path2features, f'failed_process.txt'), 'a') as f:
            f.write('\n'.join(failed) + '\n')