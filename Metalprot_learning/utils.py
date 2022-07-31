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