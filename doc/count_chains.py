"""
Author:
    Ben Orr <benjamin.orr@ucsf.edu>

Usage:
	count_chains.py
"""

import os

PATH = '20210608/_Seq_cores_reps/'

if __name__ == '__main__':

    dirlist = os.listdir(PATH)

    single_chain = []
    mult_chains = []


    for pdb in dirlist:

        open_file = open(os.path.join(PATH + pdb), "r")
        first_chain, is_mult = True, False

        for line in open_file:
            # Only look at line in pb files that correspond to protein atoms
            if line.split()[0] == 'ATOM':
                if first_chain: # compare future atoms' chains to that of the first atom
                    chain = line.split()[4]
                    first_chain = False
                elif line.split()[4] != chain:
                    mult_chains.append(pdb)
                    is_mult = True
                    break

        if not is_mult:
            single_chain.append(pdb)

        open_file.close()

    print('Single Chains:')
    print(len(single_chain))
    print('Multiple Chains:')
    print(len(mult_chains))
