"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for applying coordinate noise to input cores.
"""

#imports
from prody import ANM, AtomGroup
import numpy as np

def apply_noise(structure):
    backbone = structure.select('protein and name CA N C O CB')
    coords = backbone.getCoords()

    sample = np.random.normal(loc=0, scale=0.08, size=(backbone.numAtoms() * 3)).reshape((backbone.numAtoms(), 3))
    noised_coords = coords + sample
    
    noised_backbone = AtomGroup('protein')
    noised_backbone.setCoords(noised_coords)
    noised_backbone.setResnums(backbone.getResnums())
    noised_backbone.setResnames(backbone.getResnames())
    noised_backbone.setNames(backbone.getNames())
    noised_backbone.setChids(backbone.getChids())

    return noised_backbone

def apply_noise_ANM(structure):
    backbone = structure.select('protein and name CA N C O CB')
    coords = backbone.getCoords()

    anm = ANM('structure ANM analysis')
    anm.buildHessian(backbone)
    anm.calcModes(len(backbone.getResnums()), zeros=True)
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

    noised_coords = np.add(coords, np.dot(eigenvecs, delta_zeta))
    noised_coords = noised_coords.reshape(int(len(noised_coords) / 3), 3)
    
    noised_backbone = AtomGroup('protein')
    noised_backbone.setCoords(noised_coords)
    noised_backbone.setResnums(backbone.getResnums())
    noised_backbone.setResnames(backbone.getResnames())
    noised_backbone.setNames(backbone.getNames())

    return noised_backbone