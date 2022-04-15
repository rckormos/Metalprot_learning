"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for extracting binding core examples containing 2 to 4 ligands and writing their corresponding distance matrix/ sequence encodings.
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

def extract_cores(pdb_file: str, metal=None, selection_radius=5, no_neighbors=1):
    """Finds all putative metal binding cores in an input protein structure.

    Args:
        pdb_file (str): Path to pdb file.
        metal (str, optional): The element symbol, in all caps, of the bound metal. Defaults to None.
        selection_radius (float, optional): Defines the radius, in angstroms, in which the function looks for other coordinating residues. Defaults to 5.
        no_neighbors (int, optional): Defines the number of neighboring residues from coordinating residues to include in binding core.

    Returns:
        cores (list): List of lists of putative or known metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.
    """

    metal_sel = f'name {metal}'

    structure = parsePDB(pdb_file) #load structure
    all_resnums = structure.select('protein').getResnums()
    start_resnum = min(all_resnums) #compute residue number of first and last AA
    end_resnum = max(all_resnums)

    cores = []

    for chain_id in set(structure.getChids()): #iterate through all chains
        chain = structure.select(f'chain {chain_id}')

        metal_resnums = chain.select('hetero').select(metal_sel).getResnums()
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
    return cores

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

def remove_degenerate_cores(cores):

    if len(cores) > 1:
        unique_cores = []
        while cores:
            current_core = cores.pop()
            pairwise_rmsds = np.array([])
            for core in cores:
                rmsd = calcRMSD(current_core, core)
                pairwise_rmsds = np.append(pairwise_rmsds, rmsd)

            degenerate_core_indices = np.where(pairwise_rmsds < .3)[0]

            if len(degenerate_core_indices) > 0:
                for ind in degenerate_core_indices:
                    del core[ind]

            unique_cores.append(current_core)

    else:
        unique_cores = cores 

    return unique_cores

def writepdb(core, out_dir: str, metal: str):
    binding_core_resnums = list(set(core.select('protein').getResnums()))    
    binding_core_resnums.sort()
    pdb_id = core.getTitle()

    metal_resnum = core.select('hetero').select(f'name {metal}').getResnums()        
    filename = generate_filename(pdb_id, binding_core_resnums, '', '.pdb', metal=(metal, metal_resnum))
     
    writePDB(os.path.join(out_dir, filename), core)

def compute_labels(core, binding_core_resnums: list, metal_resnum: int):
    """Given a metal binding core, will compute the distance of all backbone atoms to metal site.

    Args:
        structure (prody.atomic.atomgroup.AtomGroup): AtomGroup of the whole structure.
        binding_core_resnums (list): List of binding core residue numbers. Note that this should be a sorted list.
        metal (int): Defines the residue number of the bound metal.

    Returns:
        distances (np.ndarray): A 1xn array containing backbone distances to metal, where n is the number of residues in the binding core. As an example, elements 0:4 contain distances between the metal and CA, C, O, and CB atoms of the binding core residue of lowest resnum.
    """

    metal_sel = core.select('hetero').select(f'resnum {metal_resnum}')
    binding_core_backbone = core.select('protein').select('name CA C CB O')
    distances = buildDistMatrix(metal_sel, binding_core_backbone)

    return distances 

def write_distance_matrices(core, output_dir: str, metal=None):
    """Generates binding core backbone distances and label files.

    Args:
        structure (prody.atomic.atomgroup.AtomGroup): AtomGroup of the whole structure.
        output_dir (str): Path to the directory to dump output files.
        binding_core_resnums (list): List of binding core residue numbers. Note that this should be a sorted list.
        metal (str, optional): Element symbol of bound metal in all caps. Defaults to None.
    """

    matrices = {}
    binding_core_resnums = list(set(core.getResnums()))
    binding_core_resnums.sort()

    for atom in ['CA', 'CB', 'C', 'N']:
        backbone = core.select('protein').select('name ' + atom)
        backbone_distances = buildDistMatrix(backbone, backbone)
        matrices[atom] = backbone_distances

    matrices['label'] = compute_labels(structure, binding_core_resnums, metal)
    matrices['resnums'] = np.array(binding_core_resnums)

    metal_resnum = core.select('hetero').select(metal).getResnums()
    filename = generate_filename(core.getTitle(), binding_core_resnums, 'distances', '.pkl', metal=(metal, metal_resnum))

    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(matrices, f)