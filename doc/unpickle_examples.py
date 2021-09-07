# unpickle_examples.py

import pickle as pkl
import numpy as np
np.set_printoptions(threshold=np.inf)
import os

# if this script is stored in Program/metal_binding_classifier/zn_data on Wynton, then it need to look back
# two folders, because pos_training_examples (my copy of it with 365 examples) is in Program
path = 'pos_training_examples/'

if __name__ == '__main__':
	
	all_mats = []

	c = 0
	for core in os.listdir(path):
		p = 0
		for permutation in os.listdir(path + '/' + core):

			perm = permutation.split('_')[-1]
			pkl_file = core + '_full_mat_' + perm + '_all_channels.pkl'

			with open(path + core + '/' + permutation + '/' + pkl_file, 'rb') as f:
				mat = pkl.load(f)

			all_mats.append(mat)

			# We only want to include the original core
			# We'll calculate permutations on-the-fly
			p+=1
			if p == 1:
				break

		# We want to include 1000 examples for now
		c+=1
		if c == 1000:
			break

	print(len(all_mats))

	with open('1000_pos_exs.pkl', 'wb') as f:
		pkl.dump(all_mats, f)
