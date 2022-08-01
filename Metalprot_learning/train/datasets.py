"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading and splitting data for model training and validation.
"""

#imports
import pandas as pd
import numpy as np
import torch

class ImageSet(torch.utils.data.Dataset):
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

def split_data(features_file: str, partitions: tuple, seed: int):
    """
    Splits data into training and test sets.
    """
    data = pd.read_pickle(features_file)
    barcodes = np.unique(np.array(data['barcodes']))
    shuffled_barcodes = np.random.shuffe(barcodes)

    train_len, test_len = round(len(data) * partitions[0]), round(len(data) * partitions[1])
    train_barcodes, test_barcodes, val_barcodes = shuffled_barcodes[0: train_len], shuffled_barcodes[train_len: train_len + test_len], shuffled_barcodes[train_len + test_len: len(data)]
    

