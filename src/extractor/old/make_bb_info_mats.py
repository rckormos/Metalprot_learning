from numpy.core.numeric import full
from numpy.lib.function_base import append
import prody as pr
import os
import numpy
import matplotlib as mpl
import pylab
from itertools import combinations, combinations_with_replacement
from docopt import docopt
import itertools
import pickle
import sys
from scipy.linalg.basic import matrix_balance
from scipy.spatial.distance import cdist

from . import ligand_database as ld
from . import features_pdb2dihe as fpdh

metal_sel = 'ion or name NI MN ZN CO CU MG FE' 


#TO DO: create artificial aa in the 4th aa.
def get_atg(full_pdb):
	'''
	prody atomgroup will be used to calc bb info.
	If the contact aa is at terminal, then the shape of the dist matrix will be < 12. So contact aa will be copied and added.
	'''

	metal = full_pdb.select(metal_sel)[0]
	contact_aas = full_pdb.select('protein and not carbon and not hydrogen and within 2.83 of resindex ' + str(metal.getResindex()))
	contact_aa_resinds = numpy.unique(contact_aas.getResindices()) 
	extention = 1 

	coords = []
	resnames = []
	names = []
	resnums = []
	resn = 1
	for resind in contact_aa_resinds:		
		ext_inds = ld.extend_res_indices([resind], full_pdb, extend =extention)

		#In some cases, the contact aa is at terminal. We can add more aa to match the shape.
		if len(ext_inds) == 2:
			if ext_inds[0] == resind:
				ext_inds.insert(0, resind)
			else:
				ext_inds.append(resind)
		if len(ext_inds) == 1:
			ext_inds.append(resind)
			ext_inds.append(resind)

		for ind in ext_inds:
			aa = full_pdb.select('resindex ' + str(ind))
			coords.extend(aa.getCoords())
			resnames.extend(aa.getResnames())
			names.extend(aa.getNames())
			resnums.extend([resn for _i in range(len(aa))])
			resn += 1


	if len(contact_aa_resinds) == 3:
		coords.extend([])
		resnames.extend([])
		names.extend([])
		resnums.extend([])

	#ag = pr.AtomGroup('-'.join([str(p) for p in per]))
	ag = pr.AtomGroup('0-1-2-3')
	ag.setCoords(coords)
	ag.setResnums(resnums)
	ag.setResnames(resnames)
	ag.setNames(names)

	return ag


def get_atgs(full_pdb, contain_metal = True):
	'''
	prody atomgroup will be used to calc bb info.
	If the contact aa is at terminal, then the shape of the dist matrix will be < 12. So contact aa will be copied and added.
	'''
	if contain_metal:
		metal = full_pdb.select(metal_sel)[0]
		contact_aas = full_pdb.select('protein and not carbon and not hydrogen and within 2.83 of resindex ' + str(metal.getResindex()))	
	else:
		#TO DO: it is not quite right here if the pdb happened to have more HIS-CYS-GLU-ASP. Skip now.
		contact_aas = full_pdb.select('resname HIS CYS GLU ASP')
		if not contact_aas and len(numpy.unique(contact_aas.getResindices())) > 4: 
			return []

	contact_aa_resinds = numpy.unique(contact_aas.getResindices()) 

	extention = 1 
	
	# TO DO: If the len of contact_ass is not 4...
	ags = []
	#for per in itertools.permutations(range(len(contact_aa_resinds))):
	for per in [range(len(contact_aa_resinds))]:
		print(per)

		coords = []
		resnames = []
		names = []
		resnums = []
		resn = 1
		for idx in per:
			resind = contact_aa_resinds[idx]

			ext_inds = ld.extend_res_indices([resind], full_pdb, extend =extention)

			#In some cases, the contact aa is at terminal. We can add more aa to match the shape.
			if len(ext_inds) == 2:
				if ext_inds[0] == resind:
					ext_inds.insert(0, resind)
				else:
					ext_inds.append(resind)
			if len(ext_inds) == 1:
				ext_inds.append(resind)
				ext_inds.append(resind)

			for ind in ext_inds:
				aa = full_pdb.select('resindex ' + str(ind))
				coords.extend(aa.getCoords())
				resnames.extend(aa.getResnames())
				names.extend(aa.getNames())
				resnums.extend([resn for _i in range(len(aa))])
				resn += 1

		ag = pr.AtomGroup('-'.join([str(p) for p in per]))
		ag.setCoords(coords)
		ag.setResnums(resnums)
		ag.setResnames(resnames)
		ag.setNames(names)

		ags.append(ag)

	return ags



def get_bb_dist_seq(core):
	'''
	If we know N CA C, The coords of CB could be calcualted. So we may not need CB coords.

	'''

	n_coords = core.select('name N').getCoords()

	c_coords = core.select('name C').getCoords()

	ca_coords = core.select('name CA').getCoords()


	n_n = cdist(n_coords, n_coords)

	c_c = cdist(c_coords, c_coords)

	ca_ca = cdist(ca_coords, ca_coords)

	cb_coords = []

	for i in range(len(n_coords)):
		Ca = ca_coords[i]
		C = c_coords[i]
		N = n_coords[i]

		b = Ca - N
		c = C - Ca
		a = numpy.cross(b, c)
		Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

		cb_coords.append(Cb)

	cb_coords = core.select('name CB').getCoords()

	cb_cb = cdist(cb_coords, cb_coords)

	return n_n, c_c, ca_ca, cb_cb


def get_dihe(ag):
	'''
	Please check features_pdb2dihe.py.
	Only the contact aa will be extracted.
	'''
	
	nres = len(ag.select('name CA'))
	print(nres)
	dist, _omega, _theta_asym, _phi_asym = fpdh.get_neighbors(ag, nres, 20.0)    

	#TO DO: extract info, only the contact aa matters?!
	omega = numpy.zeros((nres, nres))
	theta_asym = numpy.zeros((nres, nres))
	phi_asym = numpy.zeros((nres, nres))
	for i in range(1, nres, 3):
		for j in range(1, nres, 3):
			omega[i, j] = _omega[i, j]
			theta_asym[i, j] = _theta_asym[i, j]
			phi_asym[i, j] = _phi_asym[i, j]

	return omega, theta_asym, phi_asym


def get_seq_mat(ag, matrix_size = 12):

	seq = ag.select('name CA').getResnames()

	threelettercodes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',\
						'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

	seq_channels = numpy.zeros([40, matrix_size, matrix_size], dtype=int)


	for i in range(len(seq)):
		aa = seq[i]
		try:
			idx = threelettercodes.index(aa)
		except:
			print('Resname of following atom not found: {}'.format(aa))
			continue
		for j in range(len(seq)):
			seq_channels[idx][i][j] = 1 # horizontal rows of 1's in first 20 channels
			seq_channels[idx+20][j][i] = 1 # vertical columns of 1's in next 20 channels
		
	return seq_channels


def mk_full_mats(ag, matrix_size = 12):
	nres = len(ag.select('name CA'))

	n_n, c_c, ca_ca, cb_cb = get_bb_dist_seq(ag)

	omega, theta_asym, phi_asym = get_dihe(ag)

	seq_mats = get_seq_mat(ag, matrix_size)

	full_mat = numpy.zeros((47, matrix_size, matrix_size))

	# Make sure the shape of each matrix is smaller than the matrix_size.

	full_mat[0,0:n_n.shape[0], 0:n_n.shape[1]] = n_n
	full_mat[1,0:c_c.shape[0], 0:c_c.shape[1]] = c_c
	full_mat[2,0:ca_ca.shape[0], 0:ca_ca.shape[1]] = ca_ca
	full_mat[3,0:cb_cb.shape[0], 0:cb_cb.shape[1]] = cb_cb

	full_mat[4,0:omega.shape[0], 0:omega.shape[1]] = omega
	full_mat[5,0:theta_asym.shape[0], 0:theta_asym.shape[1]] = theta_asym
	full_mat[6,0:phi_asym.shape[0], 0:phi_asym.shape[1]] = phi_asym

	for i in range(7, 47):
		full_mat[i, :, :] = seq_mats[i - 7]

	return full_mat


def write_pickle_file(full_mat, pdb, ag, out_folder, tag = ''):
	"""
	Writes a pickle file containing the input numpy array into the current permutation's folder.
	Currently using this only to save the full matrix (all 46 channels).
	"""
	numpy.set_printoptions(threshold=numpy.inf)
	pdb_name = pdb.split('.')[0]
	
	pkl_file = out_folder + pdb_name + '_full_mat_' + ag.getTitle() + tag + '.pkl'

	with open(pkl_file, 'wb') as f:
		print(pkl_file)
		pickle.dump(full_mat, f)

	return


def write_dist_mat_file(mat, pdb, ag, out_folder, tag = ''):
	"""
	Writes out a file containing the distance matrix
	"""
	# output_folder = 'core_contact_maps/dist_mat_txt_folder/'

	numpy.set_printoptions(threshold=numpy.inf)

	dist_mat_file = pdb.split('.')[0]

	dist_mat_file = out_folder + dist_mat_file + '_full_mat_' + ag.getTitle() + tag  + '.txt'

	with open(dist_mat_file, 'w') as open_file:
		for i in mat:
			open_file.write(str(i) + '\n')

	return


def run_mk_bb_info_mats(workdir, out_path, mat_size = 12, top = 1000, contain_metal = True, opts = None):

	os.makedirs(out_path, exist_ok=True)

	count = 0

	errors = ''
	
	for pdb_name in os.listdir(workdir):

		if count >= top:
			break

		if '.pdb' not in pdb_name:
			continue

		pdb_file = workdir + pdb_name

		pdb = pr.parsePDB(pdb_file)

		ags = get_atgs(pdb, contain_metal)
	
		for ag in ags:
			try:
				#TO DO: currently, only consider 3 or 4 aa binding.
				if len(ag.select('name CA'))> 12 or len(ag.select('name CA')) < 7:
					print(pdb_name + ' not used. ')
					continue
				full_mat = mk_full_mats(ag, mat_size)
				write_dist_mat_file(full_mat, pdb_name, ag, out_path)
				write_pickle_file(full_mat, pdb_name, ag, out_path)

				count += 1
			except:
				print('error: ' + pdb_name)
				errors += pdb_name + '\n'

			if count >= top:
				break
	
	with open(out_path + '_error.txt', 'w') as f:
		f.write(errors)

	return



