"""
Author:
	Ben Orr <benjamin.orr@ucsf.edu>

Usage:
	make_bb_dist_mats.py dmap [options]

Options:
    -p, --pdb <pdb>             The PDB file.
    -c, --chains <chain-ids>    Comma-separated list of chain identifiers
                                (defaults to the first chain).
    -o, --output <file>         Save the plot to a file. The file format is
                                determined by the file extension.
    -m, --measure <measure>     The inter-residue distance measure [default: CA].
    -M, --mask-thresh <dist>    Hide the distances below a given threshold (in
                                angstroms).
    --plaintext                 Generate a plaintext distance/contact matrix
                                and write to stdout (recommended for
                                piping into other CLI programs).
    --asymmetric                Display the plot only in the upper-triangle.

    --title TITLE               The title of the plot (optional).
    --xlabel <label>            X-axis label [default: Coordinating Residue and its neighbors].
    --ylabel <label>            Y-axis label [default: Coordinating Residue and its neighbors].

    --font-family <font>        Font family (via matplotlib) [default: sans].
    --font-size <size>          Font size in points [default: 10].

    --width-inches <width>      Width of the plot in inches [default: 6.0].
    --height-inches <height>    Height of the plot in inches [default: 6.0].
    --dpi <dpi>                 Set the plot DPI [default: 80]

    --greyscale                 Generate a greyscale distance map.
    --no-colorbar               Hide the color bar on distance maps.
    --transparent               Set the background to transparent.
    --show-frame

    -v, --verbose               Verbose mode.

Notes:

This script imports Lei Lu's ligand_database.py 
(from https://github.com/lonelu/Metalprot/blob/2bbb51ede955dfdb744c10f73ccdf674416c453e/metalprot/apps/ligand_database.py),
but works with an edited version of the script (edited by Ben Orr)

Sticking the options from pconpy.py down here in case docopt needs them for plot_dist_mat() (which is
	essentially code copied from pconpy.py)


Currently (7.1.21), this code uses ligand_database.py (written by Lei Lu) to get the metal core. Then it calculates a
	distance matrix for all the backbone atoms in these residues. Then it uses the plotting functions from pconpy.py
	to generate and save distance maps.  It also writes out .txt files containing the distance matrices and
	information about the coordinating residues and their neighbors (that went into the distance matrix).
"""

# ligand_database.py must be in same dir as this script (doesn't need __init__.py?)


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

from . import ligand_database as ld


def get_bb_atoms(path):
	"""
	Params:
		path: Path to the pdb file

	Returns:
		bb_atoms: A list of prody Atom() objects that represent the N, CA, C, and CB's in the
					coordinating and neighboring residues.
	"""
	pdb_prody = pr.parsePDB(path) #os.path.join didn't add /'s where it was supposed to...?
	## parsePDB options:
	# backbone = parsePDB('1mkp', subset='bb')
	# calpha = parsePDB('1mkp', subset='ca')

	# 'ZN' is not a valid prody selection string, but 'name ZN' (case sensitive) is
	metal_sel = 'name ZN' # couldn't tell you why this is metal_sel. I see some cases where this is 'NI'...could it be 'ZN'? A: No
	# extend = 0

	metals = pdb_prody.select(metal_sel)
	print('Metals:')
	print(metals)

	metal_cores, mc_ext_inds = ld.get_metal_core_seq(pdb_prody, metal_sel, extend=0)
	print('Metal Cores:')
	print(metal_cores)

	neighboring_residues, nr_ext_inds = ld.get_metal_core_seq(pdb_prody, metal_sel, extend=1)
	print('Neighboring Residues:')
	print(neighboring_residues)

	# Put the different types of bb atoms into separate arrays
	n_atoms, ca_atoms, c_atoms, cb_atoms = [], [], [], []
	title = '' # adding this
	sel_pdb_prody = [] # ^ and this to avoid ref before assignment if neigh_res is empty - should change to better error handling later
	for title, sel_pdb_prody in neighboring_residues:
		for idx in nr_ext_inds:
			for atom in sel_pdb_prody:
				if atom.getResindex() == idx:
					if atom.getName() == "N":
						n_atoms.append(atom)
					elif atom.getName() == "CA":
						ca_atoms.append(atom)
						# Including this condition atom==CA condition bc I only want to add Gly to CB once, and I want to set GlyCB = GlyCA
						if atom.getResname() == 'GLY':
							new_atom = atom.copy()[0] # copy method returns an atom group, which can be indexed to return at Atom
							new_atom.setName('CB')
							cb_atoms.append(new_atom) # should this be a .copy() of the atom? # cb_atoms.append('GLY_CB_Resindex_' + str(atom.getResindex()))
					elif atom.getName() == "C":
						c_atoms.append(atom)
					elif atom.getName() == "CB":
						cb_atoms.append(atom)

	bb_atoms = (n_atoms, ca_atoms, c_atoms, cb_atoms)

	for atom_list in bb_atoms:
		for atom in atom_list:
			# if 'GLY_CB' in atom:
			# 	continue
			# else:
			print(atom.getResname())
			print(atom.getResnum())
			print(atom.getResindex())
			print(atom.getName())
			print(atom.getCoords())

	print('LEN BB Atoms')
	print(len(bb_atoms))
	print(len(sel_pdb_prody))
	print(nr_ext_inds)
	return metal_cores, bb_atoms


def calc_dist_matrix(atoms, dist_thresh=None,
		mask_thresh=None, asymmetric=False):
	"""
	Ben Orr: Adapted this from pconpy.py

	Calculate the distance matrix for a list of residues.
	This function is currently called by plot_dist_mat (7.1.21)

	Args:
		# residues: A list of ``Bio.PDB.Residue`` objects.
		atoms: A list of 'Prody.Atom()' objects.
		measure: The distance measure (optional).
		dist_thresh: (optional).
		mask_thresh: (optional).
		asymmetric: (optional).

	Returns:
		The distance matrix as a masked array.

	"""

	### We want the same size dist mat every time.
	### Using 12 x 12 as uniform mat size. 4 coordinating residues (max) x 
	###  3-residue stretches (i-1, i+1)
	# mat = numpy.zeros((len(atoms), len(atoms)), dtype="float64")
	mat = numpy.zeros((12, 12), dtype="float64")

	# after the distances are added to the upper-triangle, the nan values
	# indicate the lower matrix values, which are "empty", but can be used to
	# convey other information if needed.
	mat[:] = numpy.nan

	# Compute the upper-triangle of the underlying distance matrix.
	#
	# TODO:
	# - parallelise this over multiple processes + show benchmark results.
	# - use the lower-triangle to convey other information.
	pair_indices = combinations_with_replacement(range(len(atoms)), 2)

	for i, j in pair_indices:
		atom_a = atoms[i]
		atom_b = atoms[j]
		dist = calc_distance(atom_a, atom_b)
		mat[i,j] = dist

		if not asymmetric:
			mat[j,i] = dist

	# transpose i with j so the distances are contained only in the
	# upper-triangle.
	mat = mat.T

	## Commented this line out because I won't be using the mask_thresh functionality.
	## If I use a mask_thresh, then uncomment this.
	# mat = numpy.ma.masked_array(mat, numpy.isnan(mat))

	if dist_thresh is not None:
		mat = mat < dist_thresh

	if mask_thresh:
		mat = numpy.ma.masked_greater_equal(mat, mask_thresh)

	return mat

def calc_distance(atom_a, atom_b):
	"""
	Takes two prody Atom() objects and returns the Euclidean distance between them
	"""
	if 'BLANK_ATOM' in atom_a:
		return 0
	if 'BLANK_ATOM' in atom_b:
		return 0

	a_coords = atom_a.getCoords()
	b_coords = atom_b.getCoords()

	dist = numpy.linalg.norm(a_coords - b_coords)

	return dist


#
# pconpy.py Plotting Functions
#
def px2pt(p):
	"""Convert pixels to points.

	"""
	return p * 72. / 96.


def init_spines(hidden=[]):
	"""Initialise the plot frame, hiding the selected spines.

	Args:
		hidden: A list of spine names to hide. For example, set hidden
			to ["top", "right"] to hide both the top and right axes borders from
			the plot. All spines will be hidden if hidden is an empty list (optional).

	Returns:
		``None``.

	"""

	ax = pylab.gca()

	all_spines = ["top", "bottom", "right", "left", "polar"]

	for spine in all_spines:
		if spine in hidden:
			ax.spines[spine].set_visible(False)
		else:
			try:
				ax.spines[spine].set_visible(True)
				ax.spines[spine].set_linewidth(px2pt(0.75))
			except KeyError:
				pass

	return


def init_pylab(font_kwargs={}):
	"""Initialise and clean up the look and feel of the plotting area.

	Returns:
		``None``.

	"""

	mpl.rc("lines", linewidth=px2pt(1))
	mpl.rc("xtick", **{"direction" : "out" })
	mpl.rc("ytick", **{"direction" : "out" })
	mpl.rc("legend", frameon=False, fontsize=font_kwargs["size"], numpoints=1)
	mpl.rc("font", **font_kwargs)

	pylab.tick_params(axis="x", which="both", top="off")
	pylab.tick_params(axis="y", which="both", right="off")

	init_spines()

	return
#
# End pconpy.py Plotting Functions
#


def get_dist_mat(opts, bb_atoms):
	"""
	Takes a list of prody.Atoms and calls calc_dist_matrix
	"""
	### Commenting this out, since I don't require <dist> arg in usage
	# Distance threshold for contact maps.
	# if opts["<dist>"]:
	#     opts["<dist>"] = float(opts["<dist>"])

	if opts["--mask-thresh"]:
		opts["--mask-thresh"] = float(opts["--mask-thresh"])

	if opts["--chains"]:
		chain_ids = opts["--chains"].upper().split(",")

		# Check that pdb chain ids are alphanumeric (see:
		# http://deposit.rcsb.org/adit/).
		if not numpy.all(map(str.isalnum, chain_ids)):
			sys.stderr.write()

	### Commenting this out because of KeyError: opts["hbmap"]
	# if opts["hbmap"]:
	#     measure = "hb"
	# else:
	#     measure = opts["--measure"]

	### Commenting this out because we already have the coordinates
	# residues = get_residues(opts["--pdb"], chain_ids=chain_ids)

	#
	# Generate the underlying 2D matrix for the selected plot.
	#
	### Modified this line to just take bb_atoms as param
	if opts['--mask-thresh']:
		mat = calc_dist_matrix(bb_atoms, mask_thresh=opts['--mask-thresh'])
	else:
		mat = calc_dist_matrix(bb_atoms)

	return mat


def plot_dist_mat(opts, mat, pdb, mat_type, out_folder, curr_perm):
	"""
	This is just the 'if name=main' block from pconpy.py. Now split into
	mult functions (get_dist_mat, plot_dist_mat)
	"""

	if opts["--plaintext"]:
		if opts["cmap"] or opts["hbmap"]:
			fmt = "%d"
		else:
			fmt = "%.3f"

		numpy.savetxt(opts["--output"], mat.filled(0), fmt=fmt)
	else:
		font_kwargs = {
				"family" : opts["--font-family"],
				"size" : float(opts["--font-size"]) }

		init_pylab(font_kwargs)

		# hide all the spines i.e. no axes are drawn
		init_spines(hidden=["top", "bottom", "left", "right"])

		pylab.gcf().set_figwidth(float(opts["--width-inches"]))
		pylab.gcf().set_figheight(float(opts["--height-inches"]))

		### Changed to 12 for const. size dist maps
		pylab.xlim(0, 12) #len(bb_atoms))
		pylab.ylim(0, 12) #len(bb_atoms))

		pylab.xlabel(mat_type + ' for Coordinating Residues and their neighbors') # opts["--xlabel"])
		pylab.ylabel(mat_type + ' for Coordinating Residues and their neighbors') # opts["--ylabel"])

		ax, fig = pylab.gca(), pylab.gcf()

		if opts["--show-frame"]:
			init_spines(hidden=[])

		### Commenting this out because I only accept dmap as a option at the moment
		# if opts["cmap"] or opts["hbmap"]:
		#     map_obj = pylab.pcolormesh(mat,
		#             shading="flat", edgecolors="None", cmap=mpl.cm.Greys)
		if opts["dmap"]:
			if opts["--greyscale"]:
				cmap = mpl.cm.Greys
			else:
				cmap = mpl.cm.jet_r

			map_obj = pylab.pcolormesh(mat, shading="flat",
					edgecolors="None", cmap=cmap)

			if not opts["--no-colorbar"]:
				# draw the colour bar
				box = ax.get_position()
				pad, width = 0.02, 0.02
				cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
				cbar = pylab.colorbar(map_obj, drawedges=False, cax=cax)
				cbar.outline.set_visible(False)
				pylab.ylabel("Distance (angstroms)")
		else:
			raise NotImplementedError

		if opts["--title"] is not None:
			ax.set_title(opts["--title"], fontweight="bold")

		out_file = out_folder + pdb.split('.')[0] + '_dmap_' + ''.join([str(i) for i in curr_perm]) + '_' + mat_type + '.png'

		pylab.savefig(out_file, bbox_inches="tight",
				dpi=int(opts["--dpi"]), transparent=opts["--transparent"])

def write_res_info_file(bb_atoms, metal_cores, pdb, mat_type, out_folder, curr_perm):
	"""
	Writes out a file with information about the bb_atoms and their residues, which
	went into the distance matrix.
	"""
	pdb_name = pdb.split('.')[0]
	res_info_file = out_folder + pdb_name + '_atom_info_' + ''.join([str(i) for i in curr_perm]) + '_' + mat_type + '.txt'

	# res_info_file = pdb.split('.')[0]
	# res_info_file = output_folder + res_info_file + '_atom_info' + mat_type + '.txt'

	for title, sel_pdb_prody in metal_cores:
		print("Sel. pdb prody in Metal Cores:")
		print(str(sel_pdb_prody))

	with open(res_info_file, 'w') as open_file:
		for title, sel_pdb_prody in metal_cores:
			open_file.write(title + '\n')
			open_file.write(str(sel_pdb_prody) + '\n')
		for atom in bb_atoms:
			if 'BLANK_ATOM' in atom:
				open_file.write(atom + '\n')
			else:	
				open_file.write("%d %s %s %d\n" % (atom.getResindex(), atom.getResname(), atom.getName(), atom.getResnum()))
	open_file.close()


def write_dist_mat_file(mat, pdb, mat_type, out_folder, curr_perm, full_mat=False):
	"""
	Writes out a file containing the distance matrix
	"""
	# output_folder = 'core_contact_maps/dist_mat_txt_folder/'

	numpy.set_printoptions(threshold=numpy.inf)

	dist_mat_file = pdb.split('.')[0]
	if full_mat:
		dist_mat_file = out_folder + dist_mat_file + '_full_mat_' + ''.join([str(i) for i in curr_perm]) + '_' + mat_type + '.txt'
	else:
		dist_mat_file = out_folder + dist_mat_file + '_dist_mat_' + ''.join([str(i) for i in curr_perm]) + '_' + mat_type + '.txt'

	with open(dist_mat_file, 'w') as open_file:
		if mat_type == 'all_channels':
			for i in mat:
				for j in i:
					open_file.write(str(j) + '\n')
				open_file.write('end channel\n')
		else:
			for i in mat:
				open_file.write(str(i) + '\n')

	open_file.close()

	numpy.set_printoptions(threshold=1000)


def write_pickle_file(full_mat, pdb, mat_type, out_folder, curr_perm):
	"""
	Writes a pickle file containing the input numpy array into the current permutation's folder.
	Currently using this only to save the full matrix (all 44 channels).
	"""
	numpy.set_printoptions(threshold=numpy.inf)
	pdb_name = pdb.split('.')[0]
	pkl_file = out_folder + pdb_name + '_full_mat_' + ''.join([str(i) for i in curr_perm]) + '_' + mat_type + '.pkl'

	with open(pkl_file, 'wb') as f:
		pickle.dump(full_mat, f)


def make_permutations_of_bb_atoms(bb_atoms):
	perms = list(itertools.permutations([1, 2, 3, 4]))
	# print(perms)

	# bb_atoms = ([1,2,3,4,5,6,7,8,9,10,11,12],[12,11,10,9,8,7,6,5,4,3,2,1]) # test lists

	permuted_bb_atoms = []
	
	for perm in perms:
		new_bb_atoms = []
		for atom_list in bb_atoms:
			if len(atom_list) == 9:
				for i in range(3):
					atom_list.append('BLANK_ATOM')
			new_atom_list = [] # new_atom_list contains the new order of atoms of a single type
			new_atom_list = new_atom_list + atom_list[(perm[0]-1)*3:((perm[0]-1)*3)+3]
			new_atom_list = new_atom_list + atom_list[(perm[1]-1)*3:((perm[1]-1)*3)+3]
			new_atom_list = new_atom_list + atom_list[(perm[2]-1)*3:((perm[2]-1)*3)+3]
			new_atom_list = new_atom_list + atom_list[(perm[3]-1)*3:((perm[3]-1)*3)+3]

			print('ATOM LISTS___________________--------------------__________________----------\n\n\n\n\n\n\n\n\n\n')
			print(atom_list[(perm[0]-1)*3:((perm[0]-1)*3)+3])
			print(atom_list[(perm[1]-1)*3:((perm[1]-1)*3)+3])
			print(atom_list[(perm[2]-1)*3:((perm[2]-1)*3)+3])
			print(atom_list[(perm[3]-1)*3:((perm[3]-1)*3)+3])
			print('length atom list is:')
			print(len(atom_list))

			new_bb_atoms.append(new_atom_list)

		permuted_bb_atoms.append(new_bb_atoms)

	# print(permuted_bb_atoms)

	return permuted_bb_atoms


def add_seq_channels(channel_mat, atom_list):
	"""
	Params:
		channel_mat: 4 12 x 12 matrices that represent the distance maps between backbone atom types

		atom_list: one sublist of p_bb_atoms, which contains a list of Atom() objects of a single bb_atom type. Used to get .Resname()
				of the bb_atoms.

	Returns:
		full_mat: channel_mat with 40 more channels appended to it. The first 20 represent the sequence encoding with
				horizontal rows containing 1's. The next 20 contain the sequence encoding with vertical columns of 1's
	"""
	# List of three letter codes, organized in alphabetical order of the aa's FULL NAME
	threelettercodes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',\
						'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

	seq_channels = numpy.zeros([len(atom_list), len(atom_list), 40], dtype=int)

	for i in range(len(atom_list)):
		atom = atom_list[i]
		if atom == 'BLANK_ATOM':
			continue
		try:
			res = atom.getResname()
			print(res)
			idx = threelettercodes.index(res)
		except:
			print('Resname of following atom not found:')
			print(atom)
			continue

		for j in range(len(atom_list)):
			seq_channels[i][j][idx] = 1 # horizontal rows of 1's in first 20 channels
			seq_channels[j][i][idx+20] = 1 # vertical columns of 1's in next 20 channels

	print('\n\n\nSEQ CHANNELS\n\n\n')
	print(seq_channels)
	print(seq_channels.shape)
	print(sum(seq_channels))
	print(sum(seq_channels).shape)
	print(seq_channels[:, :, 8])
	print(seq_channels[0].shape)
	print(seq_channels[0][0].shape)
	print(seq_channels[:, :, 28])
	print(seq_channels[:, :, 2])
	print(seq_channels[:, :, 22])

	print('channel_mat')
	print(channel_mat.shape)
	print(channel_mat)

	print('\n\n\nfull mat\n\n')
	full_mat = []
	for i in range(4):
		# print(channel_mat[i, :, :])
		# print(channel_mat[i, :, :].shape) # (12, 12)
		full_mat.append(channel_mat[i, :, :])

	for i in range(40):
		print(seq_channels[:, :, i])
		print(seq_channels[:, :, i].shape)
		full_mat.append(seq_channels[:, :, i])

	full_mat = numpy.array(full_mat)
	print(full_mat.shape)

	return full_mat


'''

workdir = os.getcwd()

# Eventually, I'll want to look at _Seq_cores_reps instead of _Seq_cores, since reps is nonredundant
pdb_path = '/20210608/_Seq_cores_reps/' # '1a0b_ZN_1.pdb'

if __name__ == '__main__':
	opts = docopt(__doc__)

	two_coord_res_pdbs = []

	k = 0
	for pdb in os.listdir(workdir + pdb_path):

		# pdb = '1a0b_ZN_1.pdb'
		# If -p option is used, set pdb to that file
		if opts["--pdb"]:
			pdb = str(opts["--pdb"])

		print(pdb)

		# The absolute path to the metal core pdb file
		full_path = workdir + pdb_path + pdb

		# 7.6.21. bb_atoms is now a tuple of 4 lists. N, CA, C, and CB atoms.
		metal_cores, bb_atoms = get_bb_atoms(full_path)

		## Adding this condition to avoid error when there are only 2 coordinating residues
		if len(bb_atoms[0]) < 9:
			two_coord_res_pdbs.append(pdb)
			continue

		# 1) split bb atoms into 4 channels. Include logic for adding 0's for Gly CB
		# 2) make dist mat for each of 4 channels
		# 3) permute indices and generate dist mat's for each permutation

		# Each item in permuted_bb_atoms (a list) is a list representing a single permutation.
		# And each permutation is a list of length 4, each element representing a list of a single atom type.
		permuted_bb_atoms = make_permutations_of_bb_atoms(bb_atoms)

		perms = list(itertools.permutations([1, 2, 3, 4]))

		# for each core pdb, writing out to a folder titled: <pdb>_ZN_<num>/
		out_folder = 'pos_training_examples/' + pdb.split('.')[0] + '/'
		os.makedirs(out_folder, exist_ok=True)

		for i in range(len(permuted_bb_atoms)):

			curr_perm = perms[i]
			p_bb_atoms = permuted_bb_atoms[i]

			# for each permutation, writing out to a new folder
			p_out_folder = out_folder + pdb.split('.')[0] + '_' + ''.join([str(i) for i in curr_perm]) + '/'
			os.makedirs(p_out_folder, exist_ok=True)

			print('Current Permutation:')
			print(curr_perm)

			channel_mat = []
			# Calculate a distance matrix for atom type in the permutation
			for atom_list in p_bb_atoms:
				if len(atom_list) == 0:
					continue
				print('ATOM LIST IS:')
				print(atom_list)

				if 'BLANK_ATOM' in atom_list[0]:
					mat_type = atom_list[-1].getName()
				else:
					mat_type = atom_list[0].getName()

				mat = get_dist_mat(atom_list)
				print('MAT SHAPE IS:')
				print(mat.shape)
				channel_mat.append(mat)

				print(channel_mat)

				plot_dist_mat(mat, pdb, mat_type, p_out_folder, curr_perm)
				write_dist_mat_file(mat, pdb, mat_type, p_out_folder, curr_perm)
				write_res_info_file(atom_list, metal_cores, pdb, mat_type, p_out_folder, curr_perm)
				# clear pylab workspace for next dmap
				pylab.close()

			### 7.20.21: After I've appended the 4 atom type matrices to channel_mat, I need to add the next 40 sequence channels.
			channel_mat = numpy.array(channel_mat)
			full_mat = add_seq_channels(channel_mat, atom_list)

			write_dist_mat_file(channel_mat, pdb, 'all_channels', p_out_folder, curr_perm)
			write_dist_mat_file(full_mat, pdb, 'all_channels', p_out_folder, curr_perm, full_mat=True)
			write_pickle_file(full_mat, pdb, 'all_channels', p_out_folder, curr_perm)


		# clear pylab workspace for next dmap
		pylab.close()

		# k+=1
		# if k == 10:
		# 	with open('two_coord_res_pdbs.pkl', 'wb') as f:
		# 		pickle.dump(two_coord_res_pdbs, f)
		# 	break



		# for atom_list in bb_atoms:
		# 	print('ATOM LIST IS:')
		# 	print(atom_list)
		# 	mat_type = atom_list[0].getName()
		# 	mat = get_dist_mat(atom_list)
		# 	write_dist_mat_file(mat, pdb, mat_type)
		# 	plot_dist_mat(mat, mat_type)
		# 	write_res_info_file(atom_list, metal_cores, pdb, mat_type)
		# 	# clear pylab workspace for next dmap
		# 	pylab.close()



		# mat = get_dist_mat(bb_atoms)

		# write_dist_mat_file(mat, pdb)

		# plot_dist_mat(mat)

		# write_res_info_file(bb_atoms, metal_cores, pdb)

		# # folder to write out distance maps (.png files)
		# output_path = 'core_contact_maps/dist_mat_png_folder/'

		# # Prob a bad practice
		# opts["--output"] = output_path + pdb # + '_dist_mat.png'
'''


def run_make_bb_dist_mats(workdir, pdb_path, out_path, opts):

	# Eventually, I'll want to look at _Seq_cores_reps instead of _Seq_cores, since reps is nonredundant

	two_coord_res_pdbs = []

	k = 0
	for pdb in os.listdir(workdir + pdb_path):

		# pdb = '1a0b_ZN_1.pdb'
		# If -p option is used, set pdb to that file
		if opts["--pdb"]:
			pdb = str(opts["--pdb"])

		print(pdb)

		# The absolute path to the metal core pdb file
		full_path = workdir + pdb_path + pdb

		# 7.6.21. bb_atoms is now a tuple of 4 lists. N, CA, C, and CB atoms.
		metal_cores, bb_atoms = get_bb_atoms(full_path)

		## Adding this condition to avoid error when there are only 2 coordinating residues
		if len(bb_atoms[0]) < 9:
			two_coord_res_pdbs.append(pdb)
			continue

		# 1) split bb atoms into 4 channels. Include logic for adding 0's for Gly CB
		# 2) make dist mat for each of 4 channels
		# 3) permute indices and generate dist mat's for each permutation

		# Each item in permuted_bb_atoms (a list) is a list representing a single permutation.
		# And each permutation is a list of length 4, each element representing a list of a single atom type.
		permuted_bb_atoms = make_permutations_of_bb_atoms(bb_atoms)

		perms = list(itertools.permutations([1, 2, 3, 4]))

		# for each core pdb, writing out to a folder titled: <pdb>_ZN_<num>/
		#out_folder = 'pos_training_examples/' + pdb.split('.')[0] + '/'
		_out_folder = workdir + out_path
		os.makedirs(_out_folder, exist_ok=True)

		out_folder = _out_folder + '/' + pdb.split('.')[0] + '/'
		os.makedirs(out_folder, exist_ok=True)

		for i in range(len(permuted_bb_atoms)):

			curr_perm = perms[i]
			p_bb_atoms = permuted_bb_atoms[i]

			# for each permutation, writing out to a new folder
			p_out_folder = out_folder + pdb.split('.')[0] + '_' + ''.join([str(i) for i in curr_perm]) + '/'
			os.makedirs(p_out_folder, exist_ok=True)

			print('Current Permutation:')
			print(curr_perm)

			channel_mat = []
			# Calculate a distance matrix for atom type in the permutation
			for atom_list in p_bb_atoms:
				if len(atom_list) == 0:
					continue
				print('ATOM LIST IS:')
				print(atom_list)

				if 'BLANK_ATOM' in atom_list[0]:
					mat_type = atom_list[-1].getName()
				else:
					mat_type = atom_list[0].getName()

				mat = get_dist_mat(opts, atom_list)
				print('MAT SHAPE IS:')
				print(mat.shape)
				channel_mat.append(mat)

				print(channel_mat)

				plot_dist_mat(opts, mat, pdb, mat_type, p_out_folder, curr_perm)
				write_dist_mat_file(mat, pdb, mat_type, p_out_folder, curr_perm)
				write_res_info_file(atom_list, metal_cores, pdb, mat_type, p_out_folder, curr_perm)
				# clear pylab workspace for next dmap
				pylab.close()

			### 7.20.21: After I've appended the 4 atom type matrices to channel_mat, I need to add the next 40 sequence channels.
			channel_mat = numpy.array(channel_mat)
			full_mat = add_seq_channels(channel_mat, atom_list)

			write_dist_mat_file(channel_mat, pdb, 'all_channels', p_out_folder, curr_perm)
			write_dist_mat_file(full_mat, pdb, 'all_channels', p_out_folder, curr_perm, full_mat=True)
			write_pickle_file(full_mat, pdb, 'all_channels', p_out_folder, curr_perm)


		# clear pylab workspace for next dmap
		pylab.close()