#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Note that this script is meant to be called from within a SGE bash submission script. Variables that need to be defined by user are in all caps.
"""

#imports 
import os
import sys
from Metalprot_learning import loader
from Metalprot_learning.utils import AlignmentError, EncodingError

def distribute_tasks(path2examples: str, no_jobs: int, job_id: int):
    """
    Distributes pdb files across multiple cores for loading.
    """
    pdbs = [os.path.join(path2examples, file) for file in os.listdir(path2examples) if '.pdb' in file]
    tasks = [pdbs[i] for i in range(0, len(pdbs)) if i % no_jobs == job_id]
    return tasks

def run(pdb_file: str, permute: bool):
    """
    Wrapper function for running core loading.
    """
    try:
        print(pdb_file)
        protein = loader.Protein(pdb_file)
        cores = protein.get_cores()
        unique_cores = loader.remove_degenerate_cores(cores)
        for core in unique_cores:
            core.compute_channels()
            core.compute_labels()

            if permute:
                core.permute()

            core.write_pdb_files(path2output)
            core.write_data_files(path2output)
        failed_file_line = None

    except AlignmentError as e:
        failed_file_line = pdb_file + ' Error identifying unique cores'

    except EncodingError as e:
        failed_file_line = pdb_file + ' Unrecognized amino acid during sequence encoding'

    except:
        failed_file_line = pdb_file + ' Unknown error occured'

    return failed_file_line

if __name__ == '__main__':
    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1
    
    PATH2EXAMPLES = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/src'
    PERMUTE = True

    failed = []
    tasks = distribute_tasks(PATH2EXAMPLES, no_jobs, job_id)
    for pdb_file in tasks:
        failed_file_line = run(pdb_file, PERMUTE)
        failed.append(failed_file_line)
        
    failed = list(filter(None, failed))
    if len(failed) > 0:
        with open(os.path.join(path2output, 'failed.txt'), 'a') as f:
            f.write('\n'.join([line for line in failed]) + '\n')
