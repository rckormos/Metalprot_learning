from metalprot.apps.hull import write2pymol
import prody as pr
import itertools
import os
import numpy as np
from prody.atomic import residue
from scipy.spatial.distance import cdist
from . import utils

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

def get_contact_map(target, win_filter = None):
    '''
    calculate contact map for 2aa_sep database.
    return the ordered distance array and resindex array.
    '''
    xyzs = []
    for c in target.select('protein and name CA').getCoords():
        xyzs.append(c)
    xyzs = np.vstack(xyzs)  
    dists = cdist(xyzs, xyzs)

    dist_array = []
    id_array = []

    for i in range(len(xyzs)):
        for j in range(i+1, len(xyzs)):
            if win_filter:
                if i not in win_filter or j not in win_filter:
                    continue
            dist_array.append(dists[i, j])  
            id_array.append((i, j))
    dist_array, id_array = zip(*sorted(zip(dist_array, id_array)))
    return dist_array, id_array, dists


def get_pair(resinds, id_array_dist):
    pair_set = set()
    inds = list(range(len(resinds)))
    for i, j in itertools.combinations(inds, 2):
        x = resinds[i]
        y = resinds[j]
        # try:
        #     if id_array_dist[(x, y)] < 10:
        #         pair_set.add((i, j))
        # except:
        #     print('Error happend get_pair!')
        #     print('  key ' + str(x) + ', ' + str(y))

        if id_array_dist[(x, y)] < 10:
            pair_set.add((i, j))
    return pair_set


def get_all_cbs_depre(pdb, id_array_dist, num = 3):
    '''
    The method is not efficient
    '''
    resinds = np.unique(pdb.select('resname HIS CYS GLU ASP').getResindices())
    #print(len(resinds))
    all_cbs = []

    un_dist_dict = set()
    for cbs in itertools.combinations(resinds, num):
        #print(cbs)
        dist_ok = True
        for x,y in itertools.combinations(cbs, 2):
            if (x,y) in un_dist_dict:
                dist_ok = False
                break
            if  id_array_dist[(x, y)]> 10:
                un_dist_dict.add((x, y))
                dist_ok = False
                break
        if dist_ok:
            all_cbs.append(cbs)
    #print('all_cbs_len: {}'.format(len(all_cbs)))
    return all_cbs


def get_all_cbs(pdb, id_array_dist, nums = [3, 4]):
    resinds = np.unique(pdb.select('resname HIS CYS GLU ASP').getResindices())
    #print(len('Total res {}'.format(resinds)))
    all_cbs = []
  
    pair_set = get_pair(resinds, id_array_dist)
    inds = list(range(len(resinds)))

    paths = []
    for i in nums:     
        paths.extend(utils.combination_calc(inds, pair_set, i))
    #print(len(paths))
    for path in paths:
        cbs = [resinds[p] for p in path]
        check_pair = True
        for x, y in itertools.combinations(cbs, 2):
            # try:
            #     if id_array_dist[(x, y)] > 10:
            #         check_pair = False
            # except:
            #     print('Error happend!')
            #     print('  ' + pdb.getTitle() + ' key ' + str(x) + ', ' + str(y) + ' len ' + str(len(pdb)))
            #     check_pair = False

            if id_array_dist[(x, y)] > 10:
                check_pair = False

        if check_pair:
            all_cbs.append(cbs)  
    #print('all_cbs_len: {}'.format(len(all_cbs)))
    return all_cbs


def extract_aa_comb(pdb, id_array_dist, outdir):
    all_cbs = get_all_cbs(pdb, id_array_dist)
    count = 0
    for cbs in all_cbs:
        ### If nearby aa need to be extracted. You may need to check the connectivity. What if the aa is at the end of the chain.
        _cbs = cbs.copy()
        _cbs.extend([c-1 for c in cbs])
        _cbs.extend([c+1 for c in cbs])
        pdb_sel = pdb.select('resindex ' + ' '.join([str(c) for c in _cbs]))
        pr.writePDB(outdir + pdb.getTitle().split('.')[0] + 'count_' + str(count) + '_pos_' + '_'.join([str(c) for c in cbs]), pdb_sel)
        count += 1



def generate_decoy(pdb, outdir):
    '''
    #Test example

    import prody as pr
    import os
    from generate_decoy_core import generate_decoy
    workdir = '/mnt/e/DesignData/ligands/NoLigand/NoMetalProt_1-550/'

    outdir = workdir + 'output_1a21/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    pdb = pr.parsePDB(workdir + '1a21.pdb')

    generate_decoy(pdb, outdir)

    '''
    dist_array, id_array, dists = get_contact_map(pdb)

    id_array_dist = {}
    for i in range(len(id_array)):
        id = id_array[i]
        id_array_dist[id] = dist_array[i]

    extract_aa_comb(pdb, id_array_dist, outdir)

    return

def generate_decoy_all(workdir, outpath = 'metal_decoys/'):
    '''
    # Generate decoy metal binding cores for all decoy pdbs. 

    from generate_decoy_core import generate_decoy_all
    workdir = '/mnt/e/DesignData/ligands/NoLigand/NoMetalProt_1-550/' 
    generate_decoy_all(workdir)
    '''

    pdbs = get_all_pbd_prody(workdir)

    outdir = workdir + outpath

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    error_pdb = []
    for pdb in pdbs:
        print(pdb.getTitle())
        try:
            generate_decoy(pdb, outdir)
        except:
            error_pdb.append(pdb.getTitle())

    with open('error.pdb', 'w') as f:
        for ep in error_pdb:
            f.write(ep + '\n')

    return 

