"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for extracting 3-coordinate binding core examples.
"""

#imports
from prody import *
import numpy as np
import os

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

def writepdb(structure, binding_core_resnums: list, out_dir: str):
    binding_core_resnums.sort()
    binding_core = structure.select('resnum ' + ' '.join([str(num) for num in binding_core_resnums]))
    pdb_id = structure.getTitle()
    filename = pdb_id + '_' + '_'.join([str(num) for num in binding_core_resnums]) + '.pdb'
    writePDB(os.path.join(out_dir, filename), binding_core)

def compute_labels(core, example: str):
    """Given an input putative or known metal binding core, will compute the distance of all backbone atoms to metal site.

    Args:
        core (prody.atomic.atomgroup.AtomGroup): AtomGroup of the metal binding core.
        example (str): Defines whether the example is positive or negative. Must be a string in the set {'positive', 'negative'}.
    """


    pass

def extract_cores(pdb_file: str, output_dir: str, metal_sel=None, selection_radius=5, no_neighbors=1):
    """Finds all putative metal binding cores in an input protein structure.

    Args:
        pdb_file (str): Path to pdb file.
        output_dir (str): Defines the path to the directory to dump output files. 
        metal_sel (str, optional): Selection string for desired metal. For example, if I want to select all Zinc metals, I would use the following string: 'name ZN'. Defaults to None.
        selection_radius (float, optional): Defines the radius, in angstroms, in which the function looks for other coordinating residues. Defaults to 5.
        no_neighbors (int, optional): Defines the number of neighboring residues from coordinating residues to include in binding core.

    Returns:
        cores (list): List of lists of putative or known metal binding cores.
    """

    structure = parsePDB(pdb_file) #load structure
    all_resnums = structure.select('protein').getResnums()
    start_resnum = min(all_resnums) #compute residue number of first and last AA
    end_resnum = max(all_resnums)

    cores = []

    if metal_sel:
        metal_resnums = structure.select('hetero').select(metal_sel).getResnums()
        for num in metal_resnums:
            coordinating_resnums = list(set(structure.select(f'resname HIS GLU ASP CYS and within 2.83 of resnum {num}').getResnums())) #Play around with increasing this radius
            
            if len(coordinating_resnums) == 3:
                binding_core_resnums = []
                for number in coordinating_resnums:
                    core_fragment = get_neighbors(number, no_neighbors, start_resnum, end_resnum)
                    binding_core_resnums += core_fragment

                binding_core_resnums = list(set(binding_core_resnums))
                cores.append(binding_core_resnums) #add binding core to output
                writepdb(structure, binding_core_resnums, output_dir)

            else:
                continue

    else:
        sele = structure.select('resname GLU ASP CYS HIS') #select all potential metal binding residues
        
        res_nums = set(sele.getResnums())
        for number in res_nums:
            local_sele = structure.select(f'resname GLU ASP CYS HIS within {selection_radius} of resnum {number}')  
            local_resnums = list(set(local_sele.getResnums()))

            if len(local_resnums) == 3: #check if there are three metal binding residues in close proximity
                #TODO: in the case where there are more than three putative coordinating residues, compute CA-CB bond vectors and check if three are pointing in the same direction
                binding_core_resnums =[]
                for number in local_resnums: #identify neighboring residues compile list of binding core residue numbers
                    core_fragment = get_neighbors(number, no_neighbors, start_resnum, end_resnum)
                    binding_core_resnums += core_fragment

                binding_core_resnums = list(set(binding_core_resnums))
                cores.append(binding_core_resnums) #add binding core to output
                writepdb(structure, binding_core_resnums, output_dir)

            else:
                continue

    return cores