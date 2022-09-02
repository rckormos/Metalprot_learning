"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading metal binding cores from input PDB files of positive examples.

In brief, the extract_cores function takes in a PDB file and outputs all metal binding cores. In this context, we define a metal binding core as the coordinating residues and 
their neighbors in primary sequence. This function calls get_neighbors to identify neighbors in primary sequence. remove_degenerate_cores, as implied by the name, removes cores
found in extract_cores that are the same. For example, homomeric metalloproteins (i.e. hemoglobin) may have mutiple equivalent binding sites, which would be removed via action
of remove_degenerate_cores.
"""

import os
import numpy as np
import pandas as pd
from itertools import permutations
from scipy.spatial.distance import cdist
from pypivoter.degeneracy_cliques import enumerateCliques
from Metalprot_learning.utils import AlignmentError, EncodingError
from prody import parsePDB, AtomGroup, matchChains, buildDistMatrix, writePDB

def remove_degenerate_cores(cores: list):
    """
    Function to remove cores that are the same. For example, if the input 
    structure is a homotetramer, this function will only return one of the binding cores.
    """
    try:
        if len(cores) > 1:
            unique_cores = []
            while cores:
                ref = cores.pop() #extract last element in cores
                ref_total_atoms = ref.structure.select('protein').numAtoms()
                ref_resis = set(ref.structure.select('protein').select('name CA').getResnames())
                ref_length = len(ref_resis)

                pairwise_seqids, pairwise_overlap = np.array([]), np.array([])
                for core in cores: #iterate through all cores 
                    total_atoms = core.structure.select('protein').numAtoms()
                    resis = set(core.structure.select('protein').select('name CA').getResnames())
                    length = len(resis)

                    #if the reference and core have the same number of atoms, quantify similarity
                    if ref_total_atoms == total_atoms and ref_resis == resis and ref_length == length:    
                        try:
                            _, _, seqid, overlap = matchChains(ref.structure.select('protein'), core.structure.select('protein'))[0]
                            pairwise_seqids, pairwise_overlap = np.append(pairwise_seqids, seqid), np.append(pairwise_overlap, overlap)

                        except:
                            pairwise_seqids, pairwise_overlap = np.append(pairwise_seqids, 0), np.append(pairwise_overlap, 0)

                    else:
                        pairwise_seqids, pairwise_overlap = np.append(pairwise_seqids, 0), np.append(pairwise_overlap, 0)

                degenerate_core_indices = list(set(np.where(pairwise_seqids == 100)[0]).intersection(set(np.where(pairwise_overlap == 100)[0]))) #find all cores that are essentially the same structure

                if len(degenerate_core_indices) > 0: #remove all degenerate cores from cores list
                    cores = [cores[i] for i in range(0,len(cores)) if i not in degenerate_core_indices]

                unique_cores.append(ref) #add reference core 

        else:
            unique_cores = cores

    except:
        raise AlignmentError

    return unique_cores

def _impute_ca_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray):
    """
    Helper function for imputing CB coordinates. Returns an imputed set of CB coordinates.
    """
    B = ca - n
    C = c - ca
    A = np.cross(B, C, axis = 1)
    return (-0.58273431 * A) + (0.56802827 * B) - (0.54067466 * C)

def _compute_angles(vec1: np.ndarray, vec2: np.ndarray):
    """Helper function for computing angles between bond vectors in a vectorized fashion.

    Args:
        vec1 (np.ndarray): An nx3 array containing bond vectors.
        vec2 (np.ndarray): Another nx3 array containing bond vectors.

    Returns:
        angles (np.ndarray): The angles between the vectors. 
    """

    dot = np.sum(vec1 * vec2, axis=1)
    norm1, norm2 = np.linalg.norm(vec1, axis=1), np.linalg.norm(vec2, axis=1)
    angles = np.degrees(np.arccos(dot / (norm1 * norm2)))
    return angles

def _filter_by_angle(edge_list: np.ndarray, structure: AtomGroup, distances: np.ndarray):
    """Filters pairs of contacts based on relative orientation of Ca-Cb and Ca-Ca bond vectors. 

    Args:
        edge_list (np.ndarray): nx2 array containing pairs of contacts.
        structure (AtomGroup): AtomGroup object of input structure.
        distances (np.ndarray): Array of length n containing distance between each contact.

    Returns:
        filtered (np.ndarray): nx2 array containing filtered contacts.
    """

    #get backbone atom coordinates for all residues included in the edge list
    all_resindices = set(np.concatenate(list(edge_list)))
    coordinates = dict([(resindex, structure.select('protein').select('name C CA N').select(f'resindex {resindex}').getCoords()) for resindex in all_resindices])

    #for each pair of contacts, get coordinates for atom i and j
    n_i, n_j = np.vstack([coordinates[resindex][0].flatten() for resindex in edge_list[:,0]]), np.vstack([coordinates[resindex][0].flatten() for resindex in edge_list[:,1]])
    ca_i, ca_j = np.vstack([coordinates[resindex][1].flatten() for resindex in edge_list[:,0]]), np.vstack([coordinates[resindex][1].flatten() for resindex in edge_list[:,1]])
    c_i, c_j = np.vstack([coordinates[resindex][2].flatten() for resindex in edge_list[:,0]]), np.vstack([coordinates[resindex][2].flatten() for resindex in edge_list[:,1]])

    #compute ca-cb bond vector for atom i and j
    ca_cb_i, ca_cb_j = _impute_ca_cb(n_i, ca_i, c_i), _impute_ca_cb(n_j, ca_j, c_j)

    #compute the ca-ca bond vector between atom i and j. also compute the ca-ca/ca-cbi and ca-ca/ca-bj angles.
    ca_i_ca_j = ca_j - ca_i
    angles_i, angles_j = _compute_angles(ca_cb_i, ca_i_ca_j), _compute_angles(ca_cb_j, ca_i_ca_j)

    #filter based on angle cutoffs
    accepted = np.argwhere(distances <= 7)
    filtered_inds = np.intersect1d(np.intersect1d(np.argwhere(angles_i < 130), np.argwhere(angles_j > 30)), np.argwhere(distances > 7))
    filtered = edge_list[np.union1d(accepted, filtered_inds)]
    return filtered

class Core:
    def __init__(self, core: AtomGroup, coordinating_resis: np.ndarray, source: str):
        self.structure = core
        self.coordinating_resis = np.array([1 if i in coordinating_resis else 0 for i in core.select('name N').getResindices()])
        self.identifiers = [(i, j) for i,j in zip(core.select('protein').select('name N').getResnums(), core.select('protein').select('name N').getChids())]
        self.source = source
        metal = core.select('hetero')
        self.metal_coords = None if metal == None else metal.getCoords()[0]
        self.metal_name = None if metal == None else metal.getResnames()[0]
        self.coordination_number = len(coordinating_resis)
        self.permuted_features = None
        self.permuted_labels = None
        self.permuted_identifiers = None
        self.permuted_coordinating_resis = None
        self.channels = None 
        self.distance_matrix = None
        self.encoding = None

    def _compute_seq_channels(self, sequence: list):
        """
        Code adapted from https://github.com/lonelu/Metalprot_learning/blob/main/src/extractor/make_bb_info_mats.py
        """
        threelettercodes = {'ALA': 0 , 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'CSO': 4,'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 
                'LYS': 11, 'MET': 12, 'MSE': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'SEP': 15, 'THR': 16, 'TPO': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}

        seq_channels = np.zeros([12, 12, 40], dtype=int)
        for ind, AA in enumerate(sequence):
            if AA not in threelettercodes.keys():
                raise EncodingError

            idx = threelettercodes[AA]
            for j in range(12):
                seq_channels[ind][j][idx] = 1 # horizontal rows of 1's in first 20 channels
                seq_channels[j][ind][idx+20] = 1 # vertical columns of 1's in next 20 channels
        return np.stack([seq_channels[:, :, i] for i in range(40)], axis=0)

    def compute_channels(self):
        """
        Returns nothing, but assigns computed channels to a new attribute.
        """
        channels = np.zeros((44,12,12))
        CB = _impute_cb(self.structure.select('protein').select('name N').getCoords(), self.structure.select('protein').select('name CA').getCoords(), \
                        self.structure.select('protein').select('name C').getCoords())
        
        _distance_matrices = [buildDistMatrix(self.structure.select('protein').select(f'name {name}')) for name in ['N', 'CA', 'C']]
        _distance_matrices.append(cdist(CB, CB))
        m, n = _distance_matrices[0].shape
        channels[0:4, 0:m, 0:n] = np.stack(_distance_matrices, axis=0)        
        
        #compute sequence channels
        sequence = self.structure.select('protein').select('name N').getResnames()
        seq_channels = self._compute_seq_channels(sequence)
        channels[4:, 0:12, 0:12] = seq_channels

        self.channels = channels

    def onehotencode(self):
        seq = self.structure.select('name CA').getResnames()

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
        max_resis = 12
        padding = 20 * (max_resis - len(seq))
        encoding = np.concatenate((encoding, np.zeros((1,padding))), axis=1)
        return encoding

    def compute_features(self):
        selstr = 'name CA O C N'
        encoding = self.onehotencode()
        max_atoms = 48
        binding_core_backbone = self.structure.select('protein').select(selstr)
        full_dist_mat = buildDistMatrix(binding_core_backbone, binding_core_backbone)
        padding = max_atoms - full_dist_mat.shape[0]
        full_dist_mat = np.lib.pad(full_dist_mat, ((0,padding), (0,padding)), 'constant', constant_values=0)
        
        self.distance_matrix = full_dist_mat
        self.encoding = encoding

    def compute_labels(self, distogram=False):
        if not distogram:
            label = np.zeros(12*4)
            _label = buildDistMatrix(self.structure.select('protein').select('name N CA C O'), self.structure.select('hetero')).squeeze()
            label[0:len(_label)] = _label

        else:
            distances = buildDistMatrix(self.structure.select('protein').select('name N CA C O'), self.structure.select('hetero')).squeeze()
            bins = np.arange(0, 12.5, 0.1)
            label = np.zeros(len(48), len(bins))
            for ind, distance in enumerate(distances):
                label[ind] = np.histogram(distance, bins)[0]
        self.label = label

    def _identify_fragments(self):
        """

        """
        binding_core_identifiers = self.identifiers
        temp = binding_core_identifiers[:]
        fragments = []
        while len(temp) != 0:
            for i in range(0, len(temp)): #build up contiguous fragments by looking for adjacent resnums
                if i == 0:
                    fragment = [temp[i]]

                elif set(temp[i][1]) == set([i[1] for i in fragment]) and 1 in set([abs(temp[i][0] - j[0]) for j in fragment]):
                    fragment.append(temp[i])

            fragment = list(set(fragment)) 
            fragment.sort()
            fragment_indices = [binding_core_identifiers.index(i) for i in fragment] 
            fragments.append(fragment_indices) #build a list containing lists of indices of residues for a given fragment

            for item in fragment:
                temp.remove(item)
    
        return fragments
    
    def _permute_matrices(self, dist_mat: np.ndarray, atom_ind_permutation):
        permuted_dist_mat = np.zeros(dist_mat.shape)
        for i, atom_indi in enumerate(atom_ind_permutation):
            for j, atom_indj in enumerate(atom_ind_permutation):
                permuted_dist_mat[i,j] = dist_mat[int(atom_indi), int(atom_indj)]
        return permuted_dist_mat

    def _permute_encodings(self, encoding: np.ndarray, fragment_indices, permutation):
        permuted_encoding = np.zeros(self.encoding.shape[1])
        split_encoding = np.array_split(encoding.squeeze(), encoding.shape[1]/20)
        _permuted_encoding = sum(sum([[list(split_encoding[i]) for i in fragment_indices[j]] for j in permutation], []), []) #permute the encoding by fragment
        permuted_encoding[0:len(_permuted_encoding)] = _permuted_encoding
        return permuted_encoding
        
    def _permute_channels(self, fragment_index_permutation, sequence):
        permuted_channel =np.zeros(self.channels.shape)
        for i, I in enumerate(fragment_index_permutation):
            for j, J in enumerate(fragment_index_permutation):
                permuted_channel[0:4,I,J] = self.channels[0:4,i,j]
        permuted_channel[4:,:,:] = self._compute_seq_channels([sequence[i] for i in fragment_index_permutation])
        return permuted_channel

    def _permute_labels(self, permutation, fragment_indices, atom_indices):
        permuted_label = np.zeros(len(self.label))
        _permuted_label = np.array([])
        for i in permutation:
            frag = fragment_indices[i]
            for j in frag:
                atoms = atom_indices[j]
                for atom in atoms:
                    _permuted_label = np.append(_permuted_label, self.label[int(atom)])
        permuted_label[0:len(_permuted_label)] = _permuted_label
        return permuted_label

    def permute(self, trim=True):
        permuted_features, permuted_labels, permuted_identifiers, permuted_coordinating_resis = [], [], [], []
        fragment_indices = self._identify_fragments()
        sequence = self.structure.select('protein').select('name N').getResnames()
        fragment_permutations = permutations(list(range(0,len(fragment_indices)))) #get permutations of fragment indices
        no_atoms = len(self.structure.select('protein').select('name N CA C O'))
        atom_indices = np.split(np.linspace(0, no_atoms - 1, no_atoms), no_atoms / 4)
        for permutation in fragment_permutations:
            fragment_index_permutation = sum([fragment_indices[i] for i in permutation], [])

            if type(self.channels) == np.ndarray:
                permuted_feature = self._permute_channels(fragment_index_permutation, sequence)

            elif type(self.distance_matrix) == np.ndarray:
                atom_index_permutation = sum([list(atom_indices[i]) for i in fragment_index_permutation], []) 
                distance_matrix_permutation = self._permute_matrices(self.distance_matrix, atom_index_permutation)
                encoding_permutation = self._permute_encodings(self.encoding, fragment_indices, permutation)
                permuted_feature = np.concatenate((distance_matrix_permutation.flatten(), encoding_permutation)) if not trim else np.concatenate((distance_matrix_permutation[np.triu_indices(distance_matrix_permutation.shape[0], k=1)].flatten(), encoding_permutation))

            permuted_features.append(permuted_feature)
            permuted_labels.append(self._permute_labels(permutation, fragment_indices, atom_indices))
            permuted_identifiers.append([self.identifiers[i] for i in fragment_index_permutation])
            permuted_coordinating_resis.append([self.coordinating_resis[i] for i in fragment_index_permutation])

        self.permuted_features = permuted_features
        self.permuted_labels = permuted_labels
        self.permuted_identifiers = permuted_identifiers
        self.permuted_coordinating_resis = permuted_coordinating_resis

    def _name(self, identifiers: list, metal_identifier: str):
        return self.structure.getTitle() + '_' + '_'.join([str(tup[0]) + tup[1] for tup in identifiers]) + '_' + metal_identifier

    def write_pdb_files(self, output_dir: str):
        metal = self.structure.select('hetero')
        metal_identifier = metal.getResnames()[0] + str(metal.getResnums()[0]) + metal.getChids()[0]
        filename = self._name(self.identifiers, metal_identifier) + '.pdb.gz'
        writePDB(os.path.join(output_dir, filename), self.structure)

    def write_data_files(self, output_dir: str):
        metal = self.structure.select('hetero')
        metal_identifier = metal.getResnames()[0] + str(metal.getResnums()[0]) + metal.getChids()[0]
        filename = self._name(self.identifiers, metal_identifier) + '_features' + '.pkl'
        if self.permuted_features and self.permuted_labels and self.permuted_identifiers and self.permuted_coordinating_resis:
            df = pd.DataFrame({'channels': self.permuted_features, 'labels': self.permuted_labels, 
            'identifiers': self.permuted_identifiers, 'sources': [filename] * len(self.permuted_features), 
            'coordination_number': [self.coordination_number] * len(self.permuted_features),
            'coordinating_resis': self.permuted_coordinating_resis})
            df.to_pickle(os.path.join(output_dir, filename))

        else:
            df = pd.DataFrame({'channels': [self.channels], 'labels': [self.label], 
            'identifiers': [self.identifiers], 'sources': [filename],
            'coordination_number': [self.coordination_number],
            'coordinating_resis': [self.coordinating_resis]})
            df.to_pickle(os.path.join(output_dir, filename))

class Protein:
    def __init__(self, pdb_file: str):
        self.filepath = pdb_file
        self.structure = parsePDB(pdb_file)

    @staticmethod
    def _get_neighbors(structure: AtomGroup, coordinating_resind: int, no_neighbors: int):
        """
        Helper function for getting neighboring residues. If a terminal residue is passed, only
        one neighrbor will be returned.
        """
        chain_id = list(set(structure.select(f'resindex {coordinating_resind}').getChids()))[0]
        all_resinds = structure.select(f'chain {chain_id}').select('protein').getResindices()
        terminal = max(all_resinds)
        start = min(all_resinds)

        extend = np.array(range(-no_neighbors, no_neighbors+1))
        _core_fragment = np.full((1,len(extend)), coordinating_resind) + extend
        core_fragment = [ind for ind in list(_core_fragment[ (_core_fragment >= start) & (_core_fragment <= terminal) ]) if ind in all_resinds] #remove nonexisting neighbor residues

        return core_fragment 

    def get_cores(self, no_neighbors=1, coordination_number=(2,4)):
        """
        Extracts metal binding cores from an input protein structure. Returns a list of
        core objects.
        """
        cores = []
        metals = self.structure.select('hetero').select(f'name NI MN ZN CO CU MG FE')
        metal_resindices = metals.getResindices() 
        metal_names = metals.getNames()

        for metal_ind, name in zip(metal_resindices, metal_names):

            try: #try/except to account for solvating metal ions included for structure determination
                coordinating_resindices = list(set(self.structure.select(f'protein and not carbon and not hydrogen and within 2.83 of resindex {metal_ind}').getResindices()))

            except:
                continue
            
            if len(coordinating_resindices) <= coordination_number[1] and len(coordinating_resindices) >= coordination_number[0]:
                binding_core_resindices = []
                for ind in coordinating_resindices:
                    core_fragment = Protein._get_neighbors(self.structure, ind, no_neighbors)
                    binding_core_resindices += core_fragment

                binding_core_resindices.append(metal_ind)
                binding_core = self.structure.select('resindex ' + ' '.join([str(num) for num in binding_core_resindices]))
                cores.append(Core(binding_core, np.array(coordinating_resindices), self.filepath))

            else:
                continue
        return cores

    def get_putative_cores(self, no_neighbors=1, cutoff=15, coordination_number=(2,4)):
        putative_cores = []
        edge_list = []
        putative_coordinating_resis = self.structure.select('protein').select('name CA').select('resname HIS CYS ASP GLU')
        putative_coordinating_resindices = putative_coordinating_resis.getResindices()
        dist_mat = buildDistMatrix(putative_coordinating_resis)
        edge_weights = np.array([])
        row_indexer = 0
        for col_ind in range(len(dist_mat)):
            for row_ind in range(1+row_indexer, len(dist_mat)):
                distance = dist_mat[row_ind, col_ind]
                if distance <= cutoff:
                    edge_list.append(np.array([putative_coordinating_resindices[col_ind], putative_coordinating_resindices[row_ind]]))
                    edge_weights = np.append(edge_weights, distance)
            row_indexer += 1
        edge_list = _filter_by_angle(np.vstack(edge_list), self.structure, edge_weights)
        cliques = enumerateCliques(np.array(edge_list), coordination_number[1])[coordination_number[0]:]
        for subclique in cliques:
            for clique in subclique:
                binding_core_resindices = []
                for ind in list(clique):
                    core_fragment = Protein._get_neighbors(self.structure, ind, no_neighbors)
                    binding_core_resindices += core_fragment
                binding_core = self.structure.select('resindex ' + ' '.join([str(num) for num in binding_core_resindices]))
                putative_cores.append(Core(binding_core, clique, self.filepath))
        return putative_cores 