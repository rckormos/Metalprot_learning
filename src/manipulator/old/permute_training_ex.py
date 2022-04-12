import random
import itertools
import numpy


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
		temp_channel = numpy.zeros([12, 12])
		temp_channel[0:3,:] = channel[(perm[0]-1)*3:((perm[0]-1)*3)+3,:]
		temp_channel[3:6,:] = channel[(perm[1]-1)*3:((perm[1]-1)*3)+3,:]
		temp_channel[6:9,:] = channel[(perm[2]-1)*3:((perm[2]-1)*3)+3,:]
		temp_channel[9:12,:] = channel[(perm[3]-1)*3:((perm[3]-1)*3)+3,:]

		new_channel = numpy.zeros([12, 12])
		new_channel[:,0:3] = temp_channel[:,(perm[0]-1)*3:((perm[0]-1)*3)+3]
		new_channel[:,3:6] = temp_channel[:,(perm[1]-1)*3:((perm[1]-1)*3)+3]
		new_channel[:,6:9] = temp_channel[:,(perm[2]-1)*3:((perm[2]-1)*3)+3]
		new_channel[:,9:12] = temp_channel[:,(perm[3]-1)*3:((perm[3]-1)*3)+3]

		new_training_ex.append(new_channel)

	return numpy.array(new_training_ex)
