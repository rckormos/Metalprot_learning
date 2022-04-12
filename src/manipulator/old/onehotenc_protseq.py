"""
Author:
    Ben Orr <benjamin.orr@ucsf.edu>

Usage:
	onehotenc_protseq.py <seq>

Options:
	-s, --sequence <seq>	String of one-letter amino acid codes.
"""

from docopt import docopt
import numpy as np

# List of one letter codes, organized in alphabetical order of the aa's FULL NAME
onelettercodes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',\
					'F', 'P', 'S', 'T', 'W', 'Y', 'V']


if __name__ == '__main__':
    opts = docopt(__doc__)

    if opts["<seq>"]:
	    opts["<seq>"] = str(opts["<seq>"])

    # print(opts["<seq>"])
    # print(type(opts["<seq>"])) # <class 'str'>

    onehotembedding = np.array([])

    # iterate over the input string of one letter codes, concatenating a 20-element
    # array of zeros (and one 1) for each one letter code
    i=0
    for olc in opts["<seq>"]:
    	i+=1
    	newrow = np.zeros(20)
    	try:
    		idx = onelettercodes.index(olc)
    		newrow[idx] = 1
    	except:
    		print("%s at position %d is not recognized as an AA. Adding all 0's at this position." % (olc, i))
    	onehotembedding = np.concatenate((onehotembedding, newrow))

    print(onehotembedding)




