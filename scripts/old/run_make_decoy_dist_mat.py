import os
from  src.extractor import make_neg_dist_mats as neg
from src import config

workdir = os.getcwd()
pdb_path = '/data/metal_decoys/'
out_path = '/data/metal_decoys/out/'

opts = config.ExtractorConfig()
opts.config['dmap'] = True
opts.config['--greyscale'] = True

neg.run_get_neg_dist_mats(workdir, pdb_path, out_path, opts.config)


