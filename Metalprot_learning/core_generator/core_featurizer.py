"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for computing distance matrices, labels, and sequence encodings for metal binding cores of positive examples.

Briefly, compute_labels returns a one dimensional array containin backbone distances (in angstroms) to the metal corresponding to an input core. compute_distance_matrices returns
a distance matrix conatining all pairwise distances between backbone atoms of a given core. Finally, onehotencode returns a one dimensional array containing the one-hot encoding of
the core sequence.
"""

#imports
import numpy as np
from prody import *
from Metalprot_learning.utils import EncodingError

def compute_distance_matrices(core, noised_core, metal_name: str, no_neighbors: int, coordination_number: int, c_beta: bool):
    """Generates binding core backbone distances and label files.

    Args:
        core (prody.atomic.atomgroup.AtomGroup): Input core.
        metal_name (str): The name of the metal the core binds to.
        no_neighbors (int): Number of neighbors in primary sequence.
        coordination_number (int): User-defined upper limit for coordination number.

    Returns:
        full_dist_mat (np.ndarray): Numpy array containing pairwise distances between all atoms in the core.
        binding_core_resindices (np.ndarray): Numpy array containing resindices of binding core residues.
        binding_core_resnums (np.ndarray): Numpy array containing resnums of binding core residues.
        label (np.ndarray): A numpy array containing backbone distances to metal. 
    """

    selstr = 'name CA C N O CB' if c_beta else 'name CA O C N'

    binding_core_resnums = core.select('protein').select('name N').getResnums()
    binding_core_chids = core.select('protein').select('name N').getChids()
    binding_core_identifiers = [(binding_core_resnums[i], binding_core_chids[i]) for i in range(0,len(binding_core_resnums))]

    metal_sel = core.select('hetero').select(f'name {metal_name}')
    metal_coords = metal_sel.getCoords()[0]
    max_atoms = 5 * (coordination_number + (2*coordination_number*no_neighbors))

    binding_core_backbone = core.select('protein').select(selstr)
    full_dist_mat = buildDistMatrix(binding_core_backbone, binding_core_backbone)
    padding = max_atoms - full_dist_mat.shape[0]
    full_dist_mat = np.lib.pad(full_dist_mat, ((0,padding), (0,padding)), 'constant', constant_values=0)
    label = buildDistMatrix(metal_sel, binding_core_backbone)
    label = np.lib.pad(label, ((0,0),(0,padding)), 'constant', constant_values=0)

    noised_core_backbone = noised_core.select('protein').select(selstr)
    noised_dist_mat = buildDistMatrix(noised_core_backbone, noised_core_backbone)
    noised_dist_mat = np.lib.pad(noised_dist_mat, ((0,padding), (0,padding)), 'constant', constant_values=0)
    noised_label = buildDistMatrix(metal_sel, noised_core_backbone)
    noised_label = np.lib.pad(noised_label, ((0,0),(0,padding)), 'constant', constant_values=0)

    return full_dist_mat, label, noised_dist_mat, noised_label, metal_coords, binding_core_identifiers

def onehotencode(core, no_neighbors: int, coordinating_resis: int):
    """Adapted from Ben Orr's function from make_bb_info_mats, get_seq_mat. Generates one-hot encodings for sequences.

    Args:
        core (prody.atomic.atomgroup.AtomGroup): Input core.
        no_neighbors (int): Number of neighbors in primary sequence.
        coordination_number (int): User-defined upper limit for coordination number.

    Returns:
        encoding (np.ndarray): Numpy array containing onehot encoding of core sequence.
    """
    seq = core.select('name CA').getResnames()

    threelettercodes = {'ALA': 0 , 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'CSO': 4,'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10,
                        'LYS': 11, 'MET': 12, 'MSE': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'SEP': 15, 'THR': 16, 'TPO': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}

    encoding = np.array([[]])

    for i in range(len(seq)):
        aa = str(seq[i])

        if aa not in threelettercodes:
            raise EncodingError

        idx = threelettercodes[aa]
        one_hot = np.zeros((1,20))
        one_hot[0,idx] = 1
        encoding = np.concatenate((encoding, one_hot), axis=1)

    max_resis = coordinating_resis +  (coordinating_resis * no_neighbors * 2)
    padding = 20 * (max_resis - len(seq))
    encoding = np.concatenate((encoding, np.zeros((1,padding))), axis=1)

    return encoding

def compute_coordinate_label(core, coordinating_resindices: tuple, no_neighbors: int, coordinating_resis: int):
    residues = core.select('name CA').getResindices()
    coordinate_label = np.array([1 if residue in coordinating_resindices else 0 for residue in residues])
    padding = ((coordinating_resis * no_neighbors * 2) + coordinating_resis) - len(coordinate_label)
    coordinate_label = np.lib.pad(coordinate_label, ((0,0),(0,padding)), 'constant', constant_values=0).squeeze()
    return coordinate_label