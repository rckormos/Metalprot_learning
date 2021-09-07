"""
Author:
	Ben Orr <benjamin.orr@ucsf.edu>

Usage:
	get_coordinating_residues.py dmap [options]

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
    --xlabel <label>            X-axis label [default: Residue index].
    --ylabel <label>            Y-axis label [default: Residue index].

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
	to generate and save distance maps.
"""

# ligand_database.py must be in same dir as this script (doesn't need __init__.py?)

import prody as pr
import os
import numpy
import matplotlib as mpl
import pylab
from itertools import combinations, combinations_with_replacement
from docopt import docopt
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

	metal_cores = ld.get_metal_core_seq(pdb_prody, metal_sel, extend=0)
	print('Metal Cores:')
	print(metal_cores)

	neighboring_residues = ld.get_metal_core_seq(pdb_prody, metal_sel, extend=1)
	print('Neighboring Residues:')
	print(neighboring_residues)

	bb_atoms = []
	for title, sel_pdb_prody in neighboring_residues:
		print(str(sel_pdb_prody))
		for atom in sel_pdb_prody:
			# print(atom)
			# print(atom.getCoords())
			# print(atom.getName())
			if atom.getName() in "CA CB C N":
				bb_atoms.append(atom)
	for atom in bb_atoms:
		print(atom.getName())
		print(atom.getCoords())

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
    # mat = numpy.zeros((len(atoms), len(atoms)), dtype="float64")
    mat = numpy.zeros((48, 48), dtype="float64")

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
    mat = numpy.ma.masked_array(mat, numpy.isnan(mat))

    if dist_thresh is not None:
        mat = mat < dist_thresh

    if mask_thresh:
        mat = numpy.ma.masked_greater_equal(mat, mask_thresh)

    return mat

def calc_distance(atom_a, atom_b):
	"""
	Takes two prody Atom() objects and returns the Euclidean distance between them
	"""
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

def plot_dist_mat(opts, bb_atoms):
    """
    This is just the 'if name=main' block from pconpy.py
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

        ### Changed to 48 for const. size dist maps
        pylab.xlim(0, 48) #len(bb_atoms))
        pylab.ylim(0, 48) #len(bb_atoms))

        pylab.xlabel(opts["--xlabel"])
        pylab.ylabel(opts["--ylabel"])

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

        pylab.savefig(opts["--output"], bbox_inches="tight",
                dpi=int(opts["--dpi"]), transparent=opts["--transparent"])

def write_res_info_file(bb_atoms, metal_cores, pdb):
	"""
	Writes out a file with information about the bb_atoms and their residues, which
	went into the distance matrix.
	"""
	output_folder = 'core_contact_maps/res_info_folder/'


	res_info_file = pdb.split('.')[0]
	res_info_file = output_folder + res_info_file + '_atom_info.txt'

	print(metal_cores)

	for title, sel_pdb_prody in metal_cores:
		print(str(sel_pdb_prody))

	with open(res_info_file, 'w') as open_file:
		for title, sel_pdb_prody in metal_cores:
			open_file.write(title + '\n')
			open_file.write(str(sel_pdb_prody) + '\n')
		for atom in bb_atoms:
			open_file.write("%d %s %s %d\n" % (atom.getResindex(), atom.getResname(), atom.getName(), atom.getResnum()))
	open_file.close()

'''
workdir = os.getcwd()
pdb_path = '/20210608/_Seq_cores/' # '1a0b_ZN_1.pdb'

if __name__ == '__main__':
	opts = docopt(__doc__)

	i = 0
	for pdb in os.listdir(workdir + pdb_path):
		print(pdb)

		# pdb = '1a0b_ZN_1.pdb'
		if opts["--pdb"]:
			pdb = str(opts["--pdb"])

		output_path = 'core_contact_maps/'

		opts["--output"] = output_path + pdb + '_test_output.png'

		full_path = workdir + pdb_path + pdb

		metal_cores, bb_atoms = get_bb_atoms(full_path)

		plot_dist_mat(bb_atoms)

		write_res_info_file(bb_atoms, metal_cores, pdb)

		# clear pylab workspace for next dmap
		pylab.close()

		i+=1
		if i == 5:
			break
'''

def run_get_contact_res(workdir, pdb_path, output_path, opts):

    #opts = docopt(__doc__)

    i = 0
    for pdb in os.listdir(workdir + pdb_path):
        print(pdb)

        # pdb = '1a0b_ZN_1.pdb'
        if opts["--pdb"]:
            pdb = str(opts["--pdb"])

        output_path = 'core_contact_maps/'

        opts["--output"] = output_path + pdb + '_test_output.png'

        full_path = workdir + pdb_path + pdb

        metal_cores, bb_atoms = get_bb_atoms(full_path)

        plot_dist_mat(opts, bb_atoms)

        write_res_info_file(bb_atoms, metal_cores, pdb)

        # clear pylab workspace for next dmap
        pylab.close()

        i+=1
        if i == 5:
            break
