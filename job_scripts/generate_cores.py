#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Note that this script is meant to be called from within a SGE bash submission script. Variables that need to be defined by user are in all caps.
"""

#imports 
import os
import sys
from Metalprot_learning import loader
from Metalprot_learning.utils import AlignmentError, EncodingError, PermutationError

def distribute_tasks(path2examples: str, no_jobs: int, job_id: int):
    """
    Distributes pdb files across multiple cores for loading.
    """
    pdbs = [os.path.join(path2examples, file) for file in os.listdir(path2examples) if '.pdb' in file]
    tasks = [pdbs[i] for i in range(0, len(pdbs)) if i % no_jobs == job_id]
    return tasks

def run(pdb_file: str, permute: bool, coordination_number: tuple):
    """
    Wrapper function for running core loading.
    """
    try: 
        print(pdb_file)
        failed_file_line = None
        protein = loader.MetalloProtein(pdb_file)
        fcn_cores, cnn_cores = protein.enumerate_cores(cnn=True, fcn=True, coordination_number=coordination_number)
        unique_fcn_cores, unique_cnn_cores = loader.remove_degenerate_cores(fcn_cores), loader.remove_degenerate_cores(cnn_cores)

        for fcn_core, cnn_core in zip(unique_fcn_cores, unique_cnn_cores):
            if permute:
                fcn_core.permute(), cnn_core.permute()
                if not len(fcn_core.permuted_distance_matrices) <= 24 and len(fcn_core.permuted_encodings) <= 24 and len(cnn_core.permuted_channels) <= 24:
                    raise PermutationError        
            fcn_core.write_pdb_files(path2output)
            fcn_core.write_data_files(path2output), cnn_core.write_data_files(path2output)

    except AlignmentError as e:
        failed_file_line = pdb_file + ' Error identifying unique cores'
    except EncodingError as e:
        failed_file_line = pdb_file + ' Unrecognized amino acid during sequence encoding'
    except PermutationError as e:
        failed_file_line = pdb_file + ' Error permuting fragments'
    except:
        failed_file_line = pdb_file + ' Unknown error occured'
    return failed_file_line

if __name__ == '__main__':
    path2output = sys.argv[1]
    no_jobs = 1
    job_id = 0
    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1
    
    PATH2EXAMPLES = ''
    PERMUTE = True
    COORDINATION_NUMBER = (2,4)

    failed = []
    tasks = distribute_tasks(PATH2EXAMPLES, no_jobs, job_id)
    for pdb_file in tasks:
        failed_file_line = run(pdb_file, PERMUTE, COORDINATION_NUMBER)
        failed.append(failed_file_line)
        
    failed = list(filter(None, failed))
    if len(failed) > 0:
        with open(os.path.join(path2output, 'failed.txt'), 'a') as f:
            f.write('\n'.join([line for line in failed]) + '\n')
