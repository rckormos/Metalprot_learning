"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for extracting binding core examples and writing their corresponding distance matrix/ sequence encodings.
"""

#imports
from prody import *
import numpy as np
import os
import pickle

def get_neighbors(coordinating_resnum: int, no_neighbors: int, start_resnum: int, end_resnum: int):
    """Finds neighbors of an input coordinating residue.

    Args:
        coordinating_resnum (int): Residue number of coordinatign residue.
        no_neighbors (int): Number of neighbors desired.
        start_resnum (int): Very first residue number in input structure.
        end_resnum (int): Very last residue number in input structure.

    Returns:
        core_fragment (list): List containing resnumbers of coordinating residue and neighbors. 
    """

    extend = np.array(range(-no_neighbors, no_neighbors+1))
    _core_fragment = np.full((1,len(extend)), coordinating_resnum) + extend
    core_fragment = list(_core_fragment[ (_core_fragment > start_resnum) & (_core_fragment < end_resnum) ]) #remove nonexisting neighbor residues
    return core_fragment

def generate_filename(parent_structure_id: str, binding_core_resis: list, filetype: str, extension: str, metal: tuple):
    """Helper function for generating file names.

    Args:
        parent_structure_id (str): The pdb identifier of the parent structure.
        binding_core_resis (list): List of residue numbers that comprise the binding core.
        filetype (str): The type of file.
        extension (str): The file extension.
        metal (tuple): A tuple containing the element symbol of the metal in all caps and the residue number of said metal. 
    """

    filename = parent_structure_id + '_' + '_'.join([str(num) for num in binding_core_resis]) + metal[0] + str(metal[1]) + '_' + filetype + extension
    return filename

def writepdb(core, out_dir: str, metal: str):
    """Generates a pdb file for an input core

    Args:
        core (prody.atomic.atomgroup.AtomGroup): AtomGroup of binding core.
        out_dir (str): Path to output directory.
        metal (str): The element symbol of the bound metal in all caps.
    """
    binding_core_resnums = list(set(core.select('protein').getResnums())) #get all core residue numbers
    binding_core_resnums.sort()
    pdb_id = core.getTitle()

    metal_resnum = core.select('hetero').select(f'name {metal}').getResnums() #get the residue number of the metal for output file title         
    filename = generate_filename(pdb_id, binding_core_resnums, '', '.pdb', metal=(metal, metal_resnum))
     
    writePDB(os.path.join(out_dir, filename), core) #write core to a pdb file

def extract_cores(pdb_file: str, no_neighbors=1):
    """Finds all putative metal binding cores in an input protein structure.

    Args:
        pdb_file (str): Path to pdb file.
        no_neighbors (int, optional): Defines the number of neighboring residues from coordinating residues to include in binding core.

    Returns:
        cores (list): List of metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.
        metal_names (list): List of metal names indexed by binding core.
    """

    metal_sel = f'name NI MN ZN CO CU MG FE'

    structure = parsePDB(pdb_file) #load structure
    all_resnums = structure.select('protein').getResnums()
    start_resnum = min(all_resnums) #compute residue number of first and last AA
    end_resnum = max(all_resnums)

    cores = []

    for chain_id in set(structure.getChids()): #iterate through all chains
        chain = structure.select(f'chain {chain_id}')

        metal_resnums = chain.select('hetero').select(metal_sel).getResnums()
        metal_names = chain.select('hetero').select(metal_sel).getNames()

        for num in metal_resnums:
            coordinating_resnums = list(set(chain.select(f'resname HIS GLU ASP CYS and within 2.83 of resnum {num}').getResnums())) #Play around with increasing this radius
            
            if len(coordinating_resnums) <= 4 and len(coordinating_resnums) >= 2:
                binding_core_resnums = []
                for number in coordinating_resnums:
                    core_fragment = get_neighbors(number, no_neighbors, start_resnum, end_resnum)
                    binding_core_resnums += core_fragment

                binding_core_resnums.append(num)
                binding_core = chain.select('resnum ' + ' '.join([str(num) for num in binding_core_resnums]))
                cores.append(binding_core)

            else:
                continue
    return cores, metal_names

def remove_degenerate_cores(cores: list, metal_names: list):
    """Function to remove cores that are the same. For example, if the input structure is a homotetramer, this function will only return one of the binding cores.

    Args:
        cores (list): List of metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.

    Returns:
        unique_cores (list): List of all unique metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.
        unique_names (list): List of all metal names indexed by unique binding core.
    """

    if len(cores) > 1:
        unique_cores = []
        unique_names = []
        while cores:
            current_core = cores.pop() #extract last element in cores
            current_name = metal_names.pop()
            current_total_atoms = len(current_core.getResnums())
            pairwise_rmsds = np.array([])

            for core in cores: #iterate through all cores 
                if current_total_atoms == len(core.getResnums()): #if the current cores and core have the same number of atoms, compute RMSD
                    rmsd = calcRMSD(current_core, core)
                    pairwise_rmsds = np.append(pairwise_rmsds, rmsd)

                else:
                    continue

            degenerate_core_indices = np.where(pairwise_rmsds < .3)[0] #find all cores that are essentially the same structure

            if len(degenerate_core_indices) > 0: #remove all degenerate cores from cores list
                for ind in degenerate_core_indices:
                    del cores[ind]
                    del metal_names[ind]

            unique_cores.append(current_core) #add reference core 
            unique_names.append(current_name)

    else:
        unique_cores = cores 
        unique_names = metal_names

    return unique_cores, unique_names

def compute_labels(core, metal_name: str):
    """Given a metal binding core, will compute the distance of all backbone atoms to metal site.

    Args:
        structure (prody.atomic.atomgroup.AtomGroup): AtomGroup of the whole structure.
        binding_core_resnums (list): List of binding core residue numbers. Note that this should be a sorted list.
        metal (int): Defines the residue number of the bound metal.

    Returns:
        distances (np.ndarray): A 1xn array containing backbone distances to metal, where n is the number of residues in the binding core. As an example, elements 0:4 contain distances between the metal and CA, C, O, and CB atoms of the binding core residue of lowest resnum.
    """

    metal_sel = core.select('hetero').select(f'name {metal_name}')
    binding_core_backbone = core.select('protein').select('name CA C CB O')
    distances = buildDistMatrix(metal_sel, binding_core_backbone)

    return distances 

def write_distance_matrices(core, output_dir: str, metal_name: str):
    """Generates binding core backbone distances and label files.

    Args:
        structure (prody.atomic.atomgroup.AtomGroup): AtomGroup of the whole structure.
        output_dir (str): Path to the directory to dump output files.
        binding_core_resnums (list): List of binding core residue numbers. Note that this should be a sorted list.
        metal (str): Element symbol of bound metal in all caps.
    """

    matrices = {}
    binding_core_resnums = list(set(core.getResnums()))
    binding_core_resnums.sort()

    for atom in ['CA', 'CB', 'C', 'N']:
        backbone = core.select('protein').select('name ' + atom)
        backbone_distances = buildDistMatrix(backbone, backbone)
        matrices[atom] = backbone_distances

    matrices['label'] = compute_labels(core, binding_core_resnums, metal_name)
    matrices['resnums'] = np.array(binding_core_resnums)

    metal_resnum = core.select('hetero').select(f'name {metal_name}').getResnums()
    filename = generate_filename(core.getTitle(), binding_core_resnums, 'distances', '.pkl', (metal_name, metal_resnum))

    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(matrices, f)