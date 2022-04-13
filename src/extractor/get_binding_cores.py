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

def writepdb(structure, binding_core_resnums: list, out_dir: str, metal_resnum: None):
    binding_core_resnums.sort()
    if metal_resnum:
        binding_core_resnums.append(metal_resnum)
    binding_core = structure.select('resnum ' + ' '.join([str(num) for num in binding_core_resnums]))
    pdb_id = structure.getTitle()
    filename = pdb_id + '_' + '_'.join([str(num) for num in binding_core_resnums]) + '.pdb'
    writePDB(os.path.join(out_dir, filename), binding_core)

def compute_labels(structure, binding_core_resnums: list, metal=None):
    """Given an input putative or known metal binding core, will compute the distance of all backbone atoms to metal site.

    Args:
        structure (prody.atomic.atomgroup.AtomGroup): AtomGroup of the whole structure.
        binding_core_resnums (list): List of binding core residue numbers. Note that this should be a sorted list.
        metal (int, optional): Defines the residue number of the bound metal. Defaults to None.

    Returns:
        distances (np.ndarray): A 1xn array containing backbone distances to metal, where n is the number of residues in the binding core. As an example, elements 0:4 contain distances between the metal and CA, C, O, and CB atoms of the binding core residue of lowest resnum.
    """

    if metal:
        metal_sel = structure.select('hetero').select(f'resnum {metal}')
        binding_core_backbone = structure.select('resnum ' + ' '.join([str(num) for num in binding_core_resnums])).select('name CA C CB O')
        distances = buildDistMatrix(metal_sel, binding_core_backbone)

        return distances 

    else: #TODO: label generation for negative examples
        pass

def write_distance_matrices(structure, output_dir: str, binding_core_resnums: list, metal=None):
    """Generates binding core backbone distances and label files.

    Args:
        structure (prody.atomic.atomgroup.AtomGroup): AtomGroup of the whole structure.
        output_dir (str): Path to the directory to dump output files.
        binding_core_resnums (list): List of binding core residue numbers. Note that this should be a sorted list.
        metal (int, optional): Defines the residue number of the bound metal. Defaults to None.
    """

    matrices = {}

    generate_title = lambda x: structure.getTitle() + '_' + '_'.join([str(num) for num in binding_core_resnums]) + '_' + x
    for atom in ['CA', 'CB', 'C', 'N']:
        backbone = structure.select('resnum ' + ' '.join([str(num) for num in binding_core_resnums])).select('name ' + atom)
        backbone_distances = buildDistMatrix(backbone, backbone)
        matrices[atom] = backbone_distances

    matrices['label'] = compute_labels(structure, binding_core_resnums, metal)
    matrices['resnums'] = np.array(binding_core_resnums)
    with open(os.path.join(output_dir, generate_title('distance_matrices.pkl')), 'wb') as f:
        pickle.dump(matrices, f)

def remove_degenerate_cores():
    pass

def extract_cores(pdb_file: str, output_dir: str, metal=None, selection_radius=5, no_neighbors=1):
    """Finds all putative metal binding cores in an input protein structure.

    Args:
        pdb_file (str): Path to pdb file.
        output_dir (str): Defines the path to the directory to dump output files. 
        metal (str, optional): The element symbol, in all caps, of the bound metal. Defaults to None.
        selection_radius (float, optional): Defines the radius, in angstroms, in which the function looks for other coordinating residues. Defaults to 5.
        no_neighbors (int, optional): Defines the number of neighboring residues from coordinating residues to include in binding core.

    Returns:
        cores (list): List of lists of putative or known metal binding cores.
    """

    metal_sel = f'name {metal}'

    structure = parsePDB(pdb_file) #load structure
    all_resnums = structure.select('protein').getResnums()
    start_resnum = min(all_resnums) #compute residue number of first and last AA
    end_resnum = max(all_resnums)

    cores = []

    for chain_id in set(structure.getChids()): #iterate through all chains
        chain = structure.select(f'chain {chain_id}')

        if metal:
            metal_resnums = chain.select('hetero').select(metal_sel).getResnums()
            for num in metal_resnums:
                coordinating_resnums = list(set(chain.select(f'resname HIS GLU ASP CYS and within 2.83 of resnum {num}').getResnums())) #Play around with increasing this radius
                
                if len(coordinating_resnums) <= 4 and len(coordinating_resnums) >= 2:
                    binding_core_resnums = []
                    for number in coordinating_resnums:
                        core_fragment = get_neighbors(number, no_neighbors, start_resnum, end_resnum)
                        binding_core_resnums += core_fragment

                    binding_core = chain.select('resnum' + ' '.join([str(num) for num in binding_core_resnums]))
                    cores.append(binding_core)

                    # binding_core_resnums.sort()
                    # binding_core_resnums = list(set(binding_core_resnums))
                    # cores.append(binding_core_resnums) #add binding core to output
                    # write_distance_matrices(chain, output_dir, binding_core_resnums, num)
                    # writepdb(chain, binding_core_resnums, output_dir, num)

                else:
                    continue

        else:
            sele = chain.select('resname GLU ASP CYS HIS') #select all potential metal binding residues
            
            res_nums = set(sele.getResnums())
            for number in res_nums:
                local_sele = chain.select(f'resname GLU ASP CYS HIS within {selection_radius} of resnum {number}')  
                local_resnums = list(set(local_sele.getResnums()))

                if len(coordinating_resnums) <= 4 and len(coordinating_resnums) >= 2: #check if there are three metal binding residues in close proximity
                    #TODO: in the case where there are more than three putative coordinating residues, compute CA-CB bond vectors and check if three are pointing in the same direction
                    binding_core_resnums =[]
                    for number in local_resnums: #identify neighboring residues compile list of binding core residue numbers
                        core_fragment = get_neighbors(number, no_neighbors, start_resnum, end_resnum)
                        binding_core_resnums += core_fragment

                    binding_core = chain.select('resnum' + ' '.join([str(num) for num in binding_core_resnums]))
                    cores.append(binding_core)
                    
                    # binding_core_resnums.sort()
                    # binding_core_resnums = list(set(binding_core_resnums))
                    # cores.append(binding_core_resnums) #add binding core to output
                    # write_distance_matrices(chain, output_dir, binding_core_resnums)
                    # writepdb(chain, binding_core_resnums, output_dir)

                else:
                    continue

    return cores