"""
Author:
    This script was written by Lei Lu. Copied and edited by Ben Orr <benjamin.orr@ucsf.edu>

For original script and package from which this originates, go to:
    https://github.com/lonelu/Metalprot/blob/2bbb51ede955dfdb744c10f73ccdf674416c453e/metalprot/apps/ligand_database.py
"""


import os
import prody as pr
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt



# Basic function. 

def get_all_pbd_prody(workdir):
    '''
    find all .pdb file in a folder and load them with prody.
    return a list of pdb_prody.
    '''
    pdbs = []
    for pdb_path in os.listdir(workdir):
        if not pdb_path.endswith(".pdb"):
            continue
        try:
            pdb_prody = pr.parsePDB(workdir + pdb_path)
            pdbs.append(pdb_prody)
        except:
            print('not sure')   
    return pdbs

def writepdb(cores, outdir):
    '''
    cores = list of (pdb name, pdb_prody)
    '''

    if not os.path.exists(outdir):
            os.mkdir(outdir)
    for c in cores:
        outfile = c[0].split('.')[0]
        pr.writePDB(outdir + outfile + '.pdb', c[1])



# Prepare rcsb database. // extract seq within +-3 aa for each contact aa. 

def connectivity_filter(pdb_prody, ind, ext_ind):
    res1 = pdb_prody.select('protein and resindex ' + str(ind))
    res2 = pdb_prody.select('protein and resindex ' + str(ext_ind))
    if not res2:
        return False
    if res1[0].getResnum() - res2[0].getResnum() == ind - ext_ind and res1[0].getChid() == res2[0].getChid() and res1[0].getSegname() == res2[0].getSegname():
        return True
    return False

def extend_res_indices(inds_near_res, pdb_prody, extend = 4):
    extend_inds = []
    inds = set()
    for ind in inds_near_res:
        for i in range(-extend, extend + 1):
            the_ind = ind + i
            #if the_ind>= 0: # the_ind not in inds and connectivity_filter(pdb_prody, ind, the_ind):
            if the_ind not in inds and the_ind>= 0 and connectivity_filter(pdb_prody, ind, the_ind):         
                extend_inds.append(the_ind)
                inds.add(the_ind)
    return extend_inds

def get_metal_core_seq(pdb_prody, metal_sel, extend = 4):
    """
    Docstring written by Ben:

    Params:
        pdb_prody: a pr.parsePDB(workdir + pdb_path) object.
        metal_sel: format = 'name ZN', not sure why, but that's prob the param value I'll be passing.
        extend: number of residues to extend in each direction of the coordinating residues.

    Returns:
        A list of 'metal cores'. Each 'metal_core' is whatever this tuple is:
                (pdb_prody.getTitle() + '_' + metal + '_'+ str(count), sel_pdb_prody)
    """
    metal = metal_sel.split(' ')[-1]
    nis = pdb_prody.select(metal_sel)

    # A pdb can contain more than one NI.
    if not nis:
        return
    
    metal_cores = []
    ext_inds = []
    count = 0
    for ni in nis:
        ni_index = ni.getIndex()
        #all_near = pdb_prody.select('nitrogen or oxygen or sulfur').select('not water and within 2.83 of index ' + str(ni_index))
        all_near = pdb_prody.select('resname HIS GLU ASP CYS and within 2.83 of index ' + str(ni_index))
        if not all_near or not all_near.select('nitrogen or oxygen or sulfur') or len(all_near.select('nitrogen or oxygen or sulfur')) < 3:
            continue          
        inds = all_near.select('nitrogen or oxygen or sulfur').getResindices()
        # all_near_res = pdb_prody.select('protein and resindex ' + ' '.join([str(ind) for ind in inds]))
        # if not all_near_res or len(np.unique(all_near_res.getResindices())) < 2:
        #     continue     
        # inds_near_res =  all_near_res.getResindices()
        # ext_inds = extend_res_indices(inds_near_res, pdb_prody, extend)
        ext_inds = extend_res_indices(inds, pdb_prody, extend)
        print('EXTENDED RES INDICES')
        print(ext_inds)
        count += 1
        sel_pdb_prody = pdb_prody.select('resindex ' + ' '.join([str(ind) for ind in ext_inds]) + ' '+ str(ni.getResindex()))
        print('Metal Residue Index:')
        print(str(ni.getResindex()))
        print('Coord. Residue Indices:')
        print(' '.join([str(ind) for ind in ext_inds]))
        metal_cores.append((pdb_prody.getTitle(), sel_pdb_prody)) # + '_' + metal + '_'+ str(count), sel_pdb_prody)) # removing this part that adds _ZN_1        
    return metal_cores, ext_inds # added return ext_inds for duplicate ind's









