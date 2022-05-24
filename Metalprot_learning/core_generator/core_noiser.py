"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for applying coordinate noise to input cores.
"""

#imports
from prody import ANM
import numpy as np

def sample_from_gaussian():

    pass

def apply_noise(structure, core):
    backbone = structure.select('protein and name CA O C N')
    anm = ANM('structure ANM analysis')
    anm.buildHessian(backbone)

    


    pass