"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading metal binding cores from input PDB files of positive examples.

In brief, the extract_cores function takes in a PDB file and outputs all metal binding cores. In this context, we define a metal binding core as the coordinating residues and 
their neighbors in primary sequence. This function calls get_neighbors to identify neighbors in primary sequence. remove_degenerate_cores, as implied by the name, removes cores
found in extract_cores that are the same. For example, homomeric metalloproteins (i.e. hemoglobin) may have mutiple equivalent binding sites, which would be removed via action
of remove_degenerate_cores.
"""

#imports
from prody import parsePDB, matchChains, ANM, AtomGroup
import numpy as np
from Metalprot_learning.utils import AlignmentError

def _get_neighbors(structure, coordinating_resind: int, no_neighbors: int):
    """Helper function for extract_cores. Finds neighbors of an input coordinating residue.

    Args:
        structure (prody.AtomGroup): Structure of full-length protein.
        coordinating_resind (int): Resindex of coordinating residue.
        no_neighbors (int): Number of neighbors in primary sequence to coordinating residue.

    Returns:
        core_fragment (list): List containing resindices of coordinating residue and neighbors. 
    """

    chain_id = list(set(structure.select(f'resindex {coordinating_resind}').getChids()))[0]
    all_resinds = structure.select(f'chain {chain_id}').select('protein').getResindices()
    terminal = max(all_resinds)
    start = min(all_resinds)

    extend = np.array(range(-no_neighbors, no_neighbors+1))
    _core_fragment = np.full((1,len(extend)), coordinating_resind) + extend
    core_fragment = [ind for ind in list(_core_fragment[ (_core_fragment >= start) & (_core_fragment <= terminal) ]) if ind in all_resinds] #remove nonexisting neighbor residues

    return core_fragment

def _apply_noise(structure):
    backbone = structure.select('protein').select('name C CA N O')
    coords = backbone.getCoords()
    metals = structure.select('hetero and name NI MN ZN CO CU MG FE')

    anm = ANM('structure ANM analysis')
    anm.buildHessian(backbone)
    anm.calcModes(len(backbone.getResnums())*3, zeros=True)
    eigenvals = anm.getEigvals().round(3)
    eigenvecs = anm.getEigvecs().round(3)

    R = 1.9872 * 1e-3 #units of kcal/ mol * K
    T = 298 #units of K

    delta_zeta = np.array([])
    for val, coord in zip(eigenvals, np.dot(eigenvecs.T, coords.flatten())):

        if val == 0:
            delta_zeta = np.append(delta_zeta, 0)

        else:
            sigma = np.sqrt((R*T) / val) #eigenvalues represent spring constant with units of (kcal/mol)/A^2
            mu = coord #coordinate is assumed to be equilibrium position
            sample = np.random.normal(loc=mu, scale=sigma, size=1)
            delta_zeta = np.append(delta_zeta, sample - mu) #sample from normal distribution representing probability of observing system at a given configuration

    noised_coords = np.add(coords.flatten(), np.dot(eigenvecs, delta_zeta))
    noised_coords = noised_coords.reshape(int(len(noised_coords) / 3), 3)

    noised_backbone = AtomGroup('protein')
    noised_backbone.setCoords(np.vstack([noised_coords, metals.getCoords()]))
    noised_backbone.setResnums(np.append(backbone.getResnums(), metals.getResnums()))
    noised_backbone.setResnames(np.append(backbone.getResnames(), metals.getResnames()))
    noised_backbone.setNames(np.append(backbone.getNames(), metals.getNames()))
    noised_backbone.setChids(np.append(backbone.getChids(), metals.getChids()))

    return noised_backbone

def _parse_structure(pdb_file: str, no_neighbors: int, coordination_number: int):
    """Finds all putative metal binding cores in an input protein structure.

    Args:
        pdb_file (str): Path to input pdb file.
        no_neighors (int): Number of neighbors in primary sequence to coordinating residues.
        coordination_number (int): User-defined upper limit on coordination number.

    Returns:
        cores (list): List of metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.
        names (list): List of metal names indexed by binding core.
    """

    metal_sel = f'name NI MN ZN CO CU MG FE'
    structure = parsePDB(pdb_file) #load structure
    noised_backbone = _apply_noise(structure)

    cores, names, noised_cores = [], [], []

    metal_resindices = structure.select('hetero').select(metal_sel).getResindices()
    metal_names = structure.select('hetero').select(metal_sel).getNames()
    for metal_ind, name in zip(metal_resindices, metal_names):

        try: #try/except to account for solvating metal ions included for structure determination
            coordinating_resindices = list(set(structure.select(f'protein and not carbon and not hydrogen and within 2.83 of resindex {metal_ind}').getResindices()))

        except:
            continue
        
        if len(coordinating_resindices) <= coordination_number and len(coordinating_resindices) >= 2:
            binding_core_resindices = []
            for ind in coordinating_resindices:
                core_fragment = _get_neighbors(structure, ind, no_neighbors)
                binding_core_resindices += core_fragment

            binding_core_resindices.append(metal_ind)
            binding_core = structure.select('resindex ' + ' '.join([str(num) for num in binding_core_resindices]))
            cores.append(binding_core)
            names.append(name)

            core_resnums, core_chids = binding_core.select('name N or hetero').getResnums(), binding_core.select('name N or hetero').getChids()
            iter = 0
            for resnum, chid in zip(core_resnums, core_chids):
                if iter == 0:
                    noised_core = noised_backbone.select(f'chain {chid}').select(f'resnum {resnum}')

                else:
                    noised_core += noised_backbone.select(f'chain {chid}').select(f'resnum {resnum}')

                iter += 1
            noised_cores.append(noised_core)

        else:
            continue

    return cores, names, noised_cores

def _remove_degenerate_cores(cores: list, metal_names: list, noised_cores: list):
    """Function to remove cores that are the same. For example, if the input structure is a homotetramer, this function will only return one of the binding cores.

    Args:
        cores (list): List of metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.

    Returns:
        unique_cores (list): List of all unique metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.
        unique_names (list): List of all metal names indexed by unique binding core.
    """

    try:
        if len(cores) > 1:
            unique_cores = []
            unique_names = []
            unique_noised_cores = []

            while cores:
                current_core, current_name, current_noised_core = cores.pop(), metal_names.pop(), noised_cores.pop()

                current_total_atoms = current_core.select('protein').numAtoms()
                current_resis = set(current_core.select('protein').select('name CA').getResnames())
                current_length = len(current_resis)

                pairwise_seqids = np.array([])
                pairwise_overlap = np.array([])
                for core in cores: #iterate through all cores 
                    total_atoms = core.select('protein').numAtoms()
                    resis = set(core.select('protein').select('name CA').getResnames())
                    length = len(resis)

                    if current_total_atoms == total_atoms and current_resis == resis and current_length == length: #if the current cores and core have the same number of atoms, compute RMSD    
                        try:
                            _, _, seqid, overlap = matchChains(current_core.select('protein'), core.select('protein'))[0]
                            pairwise_seqids = np.append(pairwise_seqids, seqid)
                            pairwise_overlap = np.append(pairwise_overlap, overlap)

                        except:
                            pairwise_seqids = np.append(pairwise_seqids, 0)
                            pairwise_overlap = np.append(pairwise_overlap, 0)

                    else:
                        pairwise_seqids = np.append(pairwise_seqids, 0)
                        pairwise_overlap = np.append(pairwise_overlap, 0)
                
                degenerate_core_indices = list(set(np.where(pairwise_seqids == 100)[0]).intersection(set(np.where(pairwise_overlap == 100)[0]))) #find all cores that are essentially the same structure
                if len(degenerate_core_indices) > 0: #remove all degenerate cores from cores list
                    cores = [cores[i] for i in range(0,len(cores)) if i not in degenerate_core_indices]
                    noised_cores = [noised_cores[i] for i in range(0,len(noised_cores)) if i not in degenerate_core_indices]
                    metal_names = [metal_names[i] for i in range(0,len(metal_names)) if i not in degenerate_core_indices]

                unique_cores.append(current_core) #add reference core 
                unique_names.append(current_name)
                unique_noised_cores.append(current_noised_core)

        else:
            unique_cores = cores 
            unique_names = metal_names
            unique_noised_cores = noised_cores

    except:
        raise AlignmentError

    return unique_cores, unique_names, unique_noised_cores

def extract_positive_cores(pdb_file: str, no_neighbors: int, coordination_number: int):
    cores, names, noised_cores = _parse_structure(pdb_file, no_neighbors, coordination_number)
    unique_cores, unique_names, unique_noised_cores = _remove_degenerate_cores(cores, names, noised_cores)
    return unique_cores, unique_names, unique_noised_cores