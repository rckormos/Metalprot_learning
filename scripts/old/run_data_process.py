import os

from src.manipulator import dataprocess


workdir = os.getcwd()
in_file = '/data/metal_class/1000_pos_exs.pkl'
out_path = '/data/input/out/'

dataprocess.run(workdir, out_path, in_file, isDecoy = False)

in_file2 = '/data/metal_class/1000_decoy_exs.pkl'
dataprocess.run(workdir, out_path, in_file2, isDecoy = True)

dataprocess.npy2csv(workdir, out_path)