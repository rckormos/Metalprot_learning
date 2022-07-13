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

def compute_distance_matrices(core, noised_core, metal_name: str, no_neighbors: int, coordination_number: int):
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

    binding_core_resnums = core.select('protein').select('name N').getResnums()
    binding_core_chids = core.select('protein').select('name N').getChids()
    binding_core_identifiers = [(binding_core_resnums[i], binding_core_chids[i]) for i in range(0,len(binding_core_resnums))]
    sequence = core.select('protein').select('name N').getResnames()

    metal_sel = core.select('hetero').select(f'name {metal_name}')
    metal_coords = metal_sel.getCoords()[0]
    max_atoms = 4 * (coordination_number + (2*coordination_number*no_neighbors)) 

    core, noised_core = (_impute_cb(core), _impute_cb(noised_core)) if 'GLY' not in sequence else (core, noised_core)
    atoms = ['CA', 'C', 'N', 'CB']
    distance_matrices, labels, noised_distance_matrices, noised_labels = [], [], [], []
    for atom in atoms:
        dist_mat, label = np.zeros((max_atoms, max_atoms)), np.zeros(max_atoms)
        noised_dist_mat, noised_label = np.zeros((max_atoms, max_atoms)), np.zeros(max_atoms)

        selstr = f'name {atom}'
        selection = core.select('protein').select(selstr)
        dist_mat[0:len(selection), 0:len(selection)], label[0:len(selection)] = buildDistMatrix(selection, selection), buildDistMatrix(metal_sel, selection)

        noised_selection = noised_core.select('protein').select(selstr)
        noised_dist_mat[0:len(selection), 0:len(selection)], noised_label[0:len(selection)]  = buildDistMatrix(noised_selection, noised_selection), buildDistMatrix(metal_sel, noised_selection)

        distance_matrices.append(dist_mat), labels.append(label), noised_distance_matrices.append(noised_dist_mat), noised_labels.append(noised_label)

    return distance_matrices, labels, noised_distance_matrices, noised_labels, binding_core_identifiers, metal_coords

def compute_seq_channel(core: list):

    threelettercodes = {'ALA': 0 , 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'CSO': 4,'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 
                    'LYS': 11, 'MET': 12, 'MSE': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'SEP': 15, 'THR': 16, 'TPO': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}
    seq_channels = np.zeros([len(12), len(12), 40], dtype=int)

    sequence = core.select('protein').select('name N').getResnames()
    for ind, AA in enumerate(sequence):
        if AA not in threelettercodes.keys():
            raise EncodingError

        idx = threelettercodes[AA]
        for j in range(len(sequence)):
            seq_channels[ind][j][idx] = 1 # horizontal rows of 1's in first 20 channels
            seq_channels[j][ind][idx+20] = 1 # vertical columns of 1's in next 20 channels

    return seq_channels

def compute_coordinate_label(core, coordinating_resindices: tuple, no_neighbors: int, coordinating_resis: int):
    residues = core.select('name CA').getResindices()
    coordinate_label = np.array([1 if residue in coordinating_resindices else 0 for residue in residues])
    padding = ((coordinating_resis * no_neighbors * 2) + coordinating_resis) - len(coordinate_label)
    coordinate_label = np.concatenate((coordinate_label, np.zeros(padding)))
    return coordinate_label

def compute_all_channels(core, noised_core, metal_name: str, no_neighbors: int, coordination_number: int):
    distance_matrices, labels, noised_distance_matrices, noised_labels, binding_core_identifiers, metal_coords = compute_distance_matrices(core, noised_core, metal_name, no_neighbors, coordination_number)
