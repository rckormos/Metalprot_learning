import os
import sys
sys.path.append(r'/mnt/e/GitHub_Design/metal_binding_classifier/')
from src.extractor import make_bb_dist_mats
from src.extractor import make_bb_info_mats
from src import config

workdir = os.getcwd() + '/data/_Seq_cores_reps/'
out_path = workdir + 'out20210828/'

# opts = config.ExtractorConfig()
# opts.config['dmap'] = True
# opts.config['--greyscale'] = True

# make_bb_dist_mats.run_make_bb_dist_mats(workdir, out_path, opts.config)

make_bb_info_mats.run_mk_bb_info_mats(workdir, out_path, contain_metal = True)


workdir = os.getcwd() + '/data/_Seq_cores_reps/'
out_path = workdir + 'out20210828/'

# opts = config.ExtractorConfig()
# opts.config['dmap'] = True
# opts.config['--greyscale'] = True

# make_bb_dist_mats.run_make_bb_dist_mats(workdir, out_path, opts.config)

make_bb_info_mats.run_mk_bb_info_mats(workdir, out_path, contain_metal = False)