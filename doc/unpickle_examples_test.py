# unpickle_examples_test.py

import pickle
import numpy as np
np.set_printoptions(threshold=np.inf)
import os

# if this script is stored in Program/metal_binding_classifier/zn_data on Wynton, then it need to look back
# two folders, because pos_training_examples (my copy of it with 365 examples) is in Program
path = '../../pos_training_examples/'

if __name__ == '__main__':

	k = 0
	for core in os.listdir(path):
		for permutation in os.listdir(path + '/' + core):

			perm = permutation.split('_')[-1]
			pkl_file = core + '_full_mat_' + perm + '_all_channels.pkl'

			with open(path + core + '/' + permutation + '/' + pkl_file, 'rb') as f:
				mat = pickle.load(f)

			print(mat)

			break
		break
