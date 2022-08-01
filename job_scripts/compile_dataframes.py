"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This is a utility script for compiling pickled dataframes outputted during core loading. You don't need to run this with run_jobs.py
"""

#imports
import os
import sys
import pandas as pd

if __name__ == '__main__':
    working_dir = sys.argv[1]
    upsample = sys.argv[2] 
    pickled_files = [os.path.join(working_dir, i) for i in os.listdir(working_dir) if '.pkl' in i]
    for count, file in enumerate(pickled_files):
        df = pd.read_pickle(file)
        df['barcode'] = [count] * len(df)

        if upsample:
            #if the number of core permutations is less than the maximum, upsample
            sample_size = 24 - len(df)
            df = df.sample(n=sample_size, replace=True, random_state=69)

        compiled_df = df if count == 0 else pd.concat([compiled_df, df])
        print(f'Successfully compiled {file}')
    compiled_df.to_pickle(os.path.join(working_dir, 'compiled_data.csv'))
