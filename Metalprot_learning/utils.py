"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains utility functions and error classes.
"""

class Error(Exception):
    """Base class for other exceptions"""
    pass 

class DistMatDimError(Error):
    "Raised when dimensions of distance matrix are incorrect"
    pass

class LabelDimError(Error):
    """Raised when dimensions of label are incorrect"""
    pass

class EncodingDimError(Error):
    """Raised when dimensions of sequence encoding are incorrect"""
    pass

class EncodingError(Error):
    """Raised when at least one amino acid within the core sequence is unrecognized"""
    pass 

class PermutationError(Error):
    """Raised when permutations are done incorrectly"""
    pass