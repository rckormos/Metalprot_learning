
import os
import torch
import glob
import pandas as pd
import numpy as np
import cv2


from torch.utils.data.dataset import Dataset

# Dataset class
class image_set(Dataset):
    def __init__(self,dataframe, filepath, mode=False):
        self.dataframe = dataframe
        self.filepath = filepath
        self.mode = mode

    def __getitem__(self,index):
        seq_id = self.dataframe.iloc[index,0]
        x = read_image(self.filepath, seq_id)
      # x = x.astype(np.float32)/255.0

        if self.mode=="test":
            x = x.astype(np.float32)
            return x
        elif self.mode=="valid":
            x = x.astype(np.float32)
            y = self.dataframe.iloc[index,1]
            return x,np.array([int(y)])
        else:
            x = permute_training_ex(x)
            x = x.astype(np.float32)
            y = self.dataframe.iloc[index,1]
            return x,np.array([int(y)])

    def __len__(self):
        return self.dataframe.shape[0]


def read_image(filepath, seq_id):

    #filename = '/home/ybwu/projects/Protein/testing/mbc/input/'+ seq_id
    filename = filepath + seq_id
    image = np.load(filename)

    return image


import itertools
import random
def permute_training_ex(training_ex):
        """
        Takes an array of shape (num_channels, 12, 12), randomly shuffles each set of 3 rows
                and columns in each channel, and returns the resulting (num_channels, 12, 12) array.

        Params:
                training_ex: an array of shape (num_channels, 12, 12), representing 44 channels of
                                        a 12x12 matrix

        Returns:
                an array of shape (num_channels, 12, 12)
        """

        perms = list(itertools.permutations([1, 2, 3, 4]))
        # random.seed(0)
        perm = random.choice(perms)

        new_training_ex = []

        for channel in training_ex:
                temp_channel = np.zeros([12, 12])
                temp_channel[0:3,:] = channel[(perm[0]-1)*3:((perm[0]-1)*3)+3,:]
                temp_channel[3:6,:] = channel[(perm[1]-1)*3:((perm[1]-1)*3)+3,:]
                temp_channel[6:9,:] = channel[(perm[2]-1)*3:((perm[2]-1)*3)+3,:]
                temp_channel[9:12,:] = channel[(perm[3]-1)*3:((perm[3]-1)*3)+3,:]

                new_channel = np.zeros([12, 12])
                new_channel[:,0:3] = temp_channel[:,(perm[0]-1)*3:((perm[0]-1)*3)+3]
                new_channel[:,3:6] = temp_channel[:,(perm[1]-1)*3:((perm[1]-1)*3)+3]
                new_channel[:,6:9] = temp_channel[:,(perm[2]-1)*3:((perm[2]-1)*3)+3]
                new_channel[:,9:12] = temp_channel[:,(perm[3]-1)*3:((perm[3]-1)*3)+3]

                new_training_ex.append(new_channel)

        return np.array(new_training_ex)


from random import randrange
def random_residues():
    residue_idx = []
    for i in range(12):
        residue_idx.append(randrange(20))
        #print(residue_idx)
    residue_idx = np.array(residue_idx)

    b = np.zeros((residue_idx.size, 20))
    b[np.arange(residue_idx.size),residue_idx] = 1
    b = b.transpose()

    input_left = np.tile(b[:, None, :], (1, b.shape[1], 1))
    input_left.shape
    input_right = np.tile(b[:, :, None], (1,1,b.shape[1]))
    input_right.shape
    attention = np.concatenate([input_left, input_right], axis = 0)
 #  print("attention", attention.shape)
    return attention
