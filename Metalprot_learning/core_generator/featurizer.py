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

def _compute_ca_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray):
    """Computes imaginary alpha carbon - beta carbon bond vectors given N, CA, and C coordinates for all residues in a vectorized fashion.

    Args:
        n (np.ndarray): nx3 array containing N atom coordinates for all residues.
        ca (np.ndarray): nx3 array containing CA atom coordinates for all residues.
        c (np.ndarray): nx3 array containing C atom coordinates for all residues.

    Returns:
        ca_cb: nx3 array containing imaginary alpha carbon - beta carbon bond vectors.
    """

    #compute ca-n and ca-c bond vectors
    ca_n = np.vstack([(1 / np.linalg.norm(n - ca, axis=1))]*3).T * (n - ca)
    ca_c = np.vstack([(1 / np.linalg.norm(c - ca, axis=1))]*3).T * (c - ca)

    #using trigonometry, we can compute an imaginary ca-cb vector
    n1 = (ca_n + ca_c) * -1
    n2 = np.cross(ca_n, ca_c, axis=1)
    n1 = np.vstack([(1 / np.linalg.norm(n1, axis=1))]*3).T * n1
    n2 = np.vstack([(1 / np.linalg.norm(n2, axis=1))]*3).T * n2
    d = (1.54*np.sin(np.deg2rad(54.75))) * n2
    v = (1.54*np.cos(np.deg2rad(54.75))) * n1
    ca_cb = d+v

    return ca_cb

def _impute_cb(core):
    unique_sele = core.select('name N')
    full_sele = core.select('protein').select('name N CA C O')
    n, ca, c = core.select('name N').getCoords(), core.select('name CA').getCoords(), core.select('name C').getCoords()
    ca_cb = _compute_ca_cb(n, ca, c)

    _names, names = np.full((len(n), 1), 'CB'), full_sele.getNames()
    _resnums, resnums = unique_sele.getResnums(), full_sele.getResnums()
    _resnames, resnames = unique_sele.getResnames(), full_sele.getResnames()
    _chids, chids = unique_sele.getChids(), full_sele.getChids()
    _coords, coords = ca + ca_cb, full_sele.getCoords()

    imputed_core = AtomGroup('protein')
    imputed_core.setCoords(np.hstack((coords.reshape((len(n), 12)), _coords)).reshape((len(n)*5, 3)))
    imputed_core.setChids(np.hstack((chids.reshape((len(n), 4)), _chids.reshape((len(n),1)))).reshape((len(n)*5)))
    imputed_core.setResnames(np.hstack((resnames.reshape((len(n), 4)), _resnames.reshape((len(n),1)))).reshape((len(n)*5)))
    imputed_core.setResnums(np.hstack((resnums.reshape((len(n), 4)), _resnums.reshape((len(n),1)))).reshape((len(n)*5)))
    imputed_core.setNames(np.hstack((names.reshape((len(n), 4)), _names.reshape((len(n),1)))).reshape((len(n)*5)))

    return imputed_core

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
    sequence = core.select('protein').select('name N').getResnames()

    metal_sel = core.select('hetero').select(f'name {metal_name}')
    metal_coords = metal_sel.getCoords()[0]
    max_atoms = 5 * (coordination_number + (2*coordination_number*no_neighbors)) if c_beta else 4 * (coordination_number + (2*coordination_number*no_neighbors))

    binding_core_backbone = _impute_cb(core) if 'GLY' not in sequence and c_beta else core.select('protein').select(selstr)
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
    coordinate_label = np.concatenate((coordinate_label, np.zeros(padding)))
    return coordinate_label