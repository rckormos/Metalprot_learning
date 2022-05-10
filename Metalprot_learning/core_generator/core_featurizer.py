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

def compute_distance_matrices(core, metal_name: str, no_neighbors: int, coordination_number: int):
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

    binding_core_resindices = core.select('protein').select('name N').getResindices()
    binding_core_resnums = core.select('protein').select('name N').getResnums()

    binding_core_backbone = core.select('protein').select('name CA O C N')
    full_dist_mat = buildDistMatrix(binding_core_backbone, binding_core_backbone)

    metal_sel = core.select('hetero').select(f'name {metal_name}')
    label = buildDistMatrix(metal_sel, binding_core_backbone)
    
    max_atoms = 4 * (coordination_number + (2*coordination_number*no_neighbors))
    padding = max_atoms - full_dist_mat.shape[0]
    full_dist_mat = np.lib.pad(full_dist_mat, ((0,padding), (0,padding)), 'constant', constant_values=0)
    label = np.lib.pad(label, ((0,0),(0,padding)), 'constant', constant_values=0)

    return full_dist_mat, binding_core_resindices, binding_core_resnums, label

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
    threelettercodes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', \
                        'MET','PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

    encoding = np.array([[]])

    for i in range(len(seq)):
        aa = str(seq[i])

        if aa not in threelettercodes:
           raise EncodingError

        idx = threelettercodes.index(aa)
        one_hot = np.zeros((1,20))
        one_hot[0,idx] = 1
        encoding = np.concatenate((encoding, one_hot), axis=1)

    max_resis = coordinating_resis +  (coordinating_resis * no_neighbors * 2)
    padding = 20 * (max_resis - len(seq))
    encoding = np.concatenate((encoding, np.zeros((1,padding))), axis=1)

    return encoding