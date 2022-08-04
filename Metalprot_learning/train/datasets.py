"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading and splitting data for model training and validation.
"""

#imports
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ImageSet(Dataset):
    "Custom dataset class for CNN image data."
    def __init__(self, df: pd.DataFrame, encodings: bool):
        self.labels = np.vstack([array for array in df['labels']])
        self.observations = np.stack([channel for channel in df['channels']], axis=0) if encodings else np.stack([channel[0:4] for channel in df['channels']], axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        observation = self.observations[index]
        label = self.labels[index]
        return observation, label 

def split_data(features_file: str, path2output: str, partitions: tuple, seed: int):
    """
    Splits data into training and test sets. Returns a tuple of train, test, and validation dataframes, in that order.
    """
    data = pd.read_pickle(features_file)
    barcodes = np.unique(np.array(data['barcode']))
    np.random.seed(seed)
    np.random.shuffle(barcodes)

    train_len, test_len = round(len(barcodes) * partitions[0]), round(len(barcodes) * partitions[1])
    train_barcodes, test_barcodes, val_barcodes = barcodes[0: train_len], barcodes[train_len: train_len + test_len], barcodes[train_len + test_len: len(data)]

    with open(os.path.join(path2output, 'barcodes.json'), 'w') as f:
        json.dump({'train': train_barcodes.tolist(), 'test': test_barcodes.tolist(), 'val': val_barcodes.tolist()}, f)
        
    return data[data['barcode'].isin(train_barcodes)], data[data['barcode'].isin(test_barcodes)], data[data['barcode'].isin(val_barcodes)]