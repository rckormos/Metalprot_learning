#!/usr/bin/env python3

'''
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Wrapper script to run jobs for Metalprot_learning on wynton. Code 
adapted from Xingjie Pan's loop-helix-loop reshaping repository.

Usage:
    ./run_jobs.py <job-name> <path> <job-script> [options]

Arguments:
    <job-name> 
        Job name of user's choosing.

    <path>
        Path to output directory to dump job output files and logs.

    <job-script>
        Job script for the analysis.

Options:
    --job-distributor=<JD>, -d=<JD>  [default: sequential]
        Job distributor that runs the jobs.

    --num-jobs=<NJ>, -n=<NJ>
        Number of jobs for parallel run.
'''

import docopt
import shutil
import os
import subprocess

def run_SGE(job_name: str, num_jobs: int, path: str, job_script: str, time='00:05:00', mem_free_GB=3, scratch_space_GB=1, keep_job_output_path=True):

    """Runs SGE job on UCSF Wynton cluster.

    Args:
        num_jobs (int): Number of jobs.
        path (str): Path to output directory to store job logs and output files.
        job_script (str): Path to the script to be submitted.
        time (str, optional): Time alloted for each task. Defaults to '5:00:00'.
        mem_free_GB (int, optional): Defaults to 3.
        scratch_space_GB (int, optional): Defaults to 1.
        keep_job_output_path (bool, optional): Defaults to True.
    """

    job_output_path = os.path.join(path, "job_outputs")
    if not keep_job_output_path and os.path.exists(job_output_path): 
        shutil.rmtree(job_output_path)

    if not os.path.exists(job_output_path):
        os.mkdir(job_output_path)

    qsub_command = ['qsub',
                        '-cwd'] \
                     + ['-N', job_name,
                        '-t', '1-{0}'.format(num_jobs),
                        '-l', 'h_rt={0}'.format(time),
                        '-o', job_output_path,
                        '-e', job_output_path,
                        './activate_env.sh',
                        job_script,
                        path] \
                        + [num_jobs]

    subprocess.check_call(qsub_command)

if __name__ == '__main__':
    
    arguments = docopt.docopt(__doc__)

    if arguments['--job-distributor'] == 'SGE':
        num_jobs = arguments['--num-jobs'] if arguments['--num-jobs'] else 1
        path = arguments['<path>']
        job_script = arguments['<job-script>']
        job_name = arguments['<job-name>']
        
        run_SGE(job_name, num_jobs, path, job_script)

    else:
        raise IOError('Unknown job distributor: {0}'.format(arguments['--job-distributor']))