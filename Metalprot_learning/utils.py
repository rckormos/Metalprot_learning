"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains utility functions and error classes.
"""

class Error(Exception):
    """Base class for other exceptions"""
    pass 

class AlignmentError(Error):
    """Raised when identification of unique cores fails"""
    pass

class CoreLoadingError(Error):
    """Raised when cores are not loaded successfully"""
    pass

class NoCoresError(Error):
    """Raised when no cores are found"""
    pass

class NoisingError(Error):
    """Raised when core noising was done improperly"""
    pass

class FeaturizationError(Error):
    """Raised when featurization fails"""
    pass

class EncodingError(Error):
    """Raised when at least one amino acid within the core sequence is unrecognized"""
    pass 

class PermutationError(Error):
    """Raised when permutations are done incorrectly"""
    pass 