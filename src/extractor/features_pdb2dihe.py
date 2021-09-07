
import io
import os
import numpy as np
from prody.utilities.catchall import getCoords
import scipy
import scipy.spatial
import string
import prody as pr

'''
The file is provided by Yibing Wu and modified by Lei Lu to remove Rosetta dependence.

The functions are used to calculate pairwise amino acids dihedral angles. 
For more details, please check <Improved protein structure prediction using predicted interresidue orientations> Figure 1.
(www.pnas.org/cgi/doi/10.1073/pnas.1914677117)

'''

def get_dihedrals(a, b, c, d):
    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1, axis=-1)[:,None]
    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1
    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)
    return np.arctan2(y, x)

def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]
    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]
    x = np.sum(v*w, axis=1)
    return np.arccos(x)

def get_neighbors(pose, nres, dmax):

    N = np.stack([r.select('name N').getCoords()[0] for r in pose.iterResidues() if r.select('aminoacid')])
    Ca = np.stack([r.select('name CA').getCoords()[0] for r in pose.iterResidues() if r.select('aminoacid')])
    C = np.stack([r.select('name C').getCoords()[0] for r in pose.iterResidues() if r.select('aminoacid')])


    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    
    # fast neighbors search
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)
    
    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]
    
    # Cb-Cb distance matrix
    dist6d = np.zeros((nres, nres))
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
    
    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    
    return dist6d, omega6d, theta6d, phi6d

def cal_phi(r):
    try:
        return pr.calcPhi(r)
    except:
        return 0

def cal_psi(r):
    try:
        return pr.calcPsi(r)
    except:
        return 0

def parse_pdb_6d(filename):

    # load PDB
    p = pr.parsePDB(filename)
    nres = len(p.select('name CA'))
    
    # backbone (phi,psi)
    phi = np.array(np.deg2rad([cal_phi(r) for r in p.iterResidues() if r.select('aminoacid')])).astype(np.float32)
    psi = np.array(np.deg2rad([cal_psi(r) for r in p.iterResidues() if r.select('aminoacid')])).astype(np.float32)

    # Ca xyz[] coordinates
    #xyz = np.array([r.select('name CA').getCoords()[0] for r in p.iterResidues() if r.select('aminoacid')]).astype(np.float16)
    
    # 6D coordinates
    dist, omega, theta_asym, phi_asym = get_neighbors(p, nres, 20.0)    
    
    return {'phi' : phi, 
            'psi' : psi,
            'omega6d' : omega, 
            'theta6d' : theta_asym, 
            'phi6d' : phi_asym}


'''
# Running example
filename = '/mnt/e/GitHub_Design/metal_binding_classifier/temp/9gaa.pdb'
features = parse_pdb_6d(filename)
#print(features)
np.savez_compressed('9gaa_1_A_repeat.npz', 
        phi=features['phi'],
        psi=features['psi'], 
        omega6d=features['omega6d'], 
        theta6d=features['theta6d'], 
        phi6d=features['phi6d'], 
        )
'''