import os
import pickle as pkl

workdir = '/wynton/home/degradolab/lonelu/GitHub_Design/metal_binding_classifier/data/_Seq_cores_reps/out_20210829/'

#workdir = '/wynton/home/degradolab/lonelu/GitHub_Design/metal_binding_classifier/data/metal_decoys/out_decoy_20210829/'

outdir = '/wynton/home/degradolab/lonelu/GitHub_Design/metal_binding_classifier/data/metal_class/'
all_mats = []
count = 0
for file in os.listdir(workdir):
    if not '.pkl' in file:
        continue
    with open(workdir + file, 'rb') as f:
        mat = pkl.load(f)
    all_mats.append(mat)
    count +=1

pkl_dump_name = str(count) + '_all_target_exs_20210829.pkl'

with open(outdir + pkl_dump_name, 'wb') as f:
    pkl.dump(all_mats, f)
    