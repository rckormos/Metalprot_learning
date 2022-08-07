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
from Metalprot_learning.utils import AlignmentError, EncodingError
from prody import parsePDB, AtomGroup, matchChains, buildDistMatrix, writePDB

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

def _impute_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray):
    """
    Helper function for imputing CB coordinates. Returns an imputed set of CB coordinates.
    """
    B = ca - n
    C = c - ca
    A = np.cross(B, C, axis = 1)
    return (-0.58273431 * A) + (0.56802827 * B) - (0.54067466 * C) + ca

class Core:
    def __init__(self, core: AtomGroup, coordinating_resis: np.ndarray, source: str):
        self.structure = core
        self.coordinating_resis = coordinating_resis
        self.identifiers = [(i, j) for i,j in zip(core.select('protein').select('name N').getResnums(), core.select('protein').select('name N').getChids())]
        self.source = source
        self.metal_coords = core.select('hetero').getCoords()[0]
        self.metal_name = core.select('hetero').getResnames()[0]
        self.coordination_number = len(coordinating_resis)
        self.permuted_channels = None
        self.permuted_labels = None
        self.permuted_identifiers = None
        self.permuted_coordinating_resis = None

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

    def compute_labels(self):
        label = np.zeros(12*4)
        _label = buildDistMatrix(self.structure.select('protein').select('name N CA C O'), self.structure.select('hetero')).squeeze()
        label[0:len(_label)] = _label
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

    def permute(self):
        permuted_channels, permuted_labels, permuted_identifiers, permuted_coordinating_resis = [], [], [], []
        fragment_indices = self._identify_fragments()
        sequence = self.structure.select('protein').select('name N').getResnames()
        fragment_permutations = permutations(list(range(0,len(fragment_indices)))) #get permutations of fragment indices
        for permutation in fragment_permutations:
            permuted_channel, permuted_label = np.zeros(self.channels.shape), np.zeros(len(self.label))
            fragment_index_permutation = sum([fragment_indices[i] for i in permutation], [])

            for i, I in enumerate(fragment_index_permutation):
                for j, J in enumerate(fragment_index_permutation):
                    permuted_channel[0:4,I,J] = self.channels[0:4,i,j]

            permuted_channel[4:,:,:] = self._compute_seq_channels([sequence[i] for i in fragment_index_permutation])

            no_atoms = len(self.structure.select('protein').select('name N CA C O'))
            atom_indices = np.split(np.linspace(0, no_atoms - 1, no_atoms), no_atoms / 4)
            _permuted_label = np.array([])
            for i in permutation:
                frag = fragment_indices[i]
                for j in frag:
                    atoms = atom_indices[j]
                    for atom in atoms:
                        _permuted_label = np.append(_permuted_label, self.label[int(atom)])            
            permuted_label[0:len(_permuted_label)] = _permuted_label

            permuted_channels.append(permuted_channel)
            permuted_labels.append(permuted_label)
            permuted_identifiers.append([self.identifiers[i] for i in fragment_index_permutation])
            permuted_coordinating_resis.append([self.coordinating_resis[i] for i in fragment_index_permutation])

        self.permuted_channels = permuted_channels
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
        filename = self._name(self.identifiers, metal_identifier) + '.pkl'
        if self.permuted_channels and self.permuted_labels and self.permuted_identifiers and self.permuted_coordinating_resis:
            df = pd.DataFrame({'channels': self.permuted_channels, 'labels': self.permuted_labels, 
            'identifiers': self.permuted_identifiers, 'sources': [filename] * len(self.permuted_channels), 
            'coordination_number': [self.coordination_number] * len(self.permuted_channels),
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
                    core_fragment = _get_neighbors(self.structure, ind, no_neighbors)
                    binding_core_resindices += core_fragment

                binding_core_resindices.append(metal_ind)
                binding_core = self.structure.select('resindex ' + ' '.join([str(num) for num in binding_core_resindices]))
                cores.append(Core(binding_core, np.array(coordinating_resindices), self.filepath))

            else:
                continue
        return cores