import os
import numpy as np
import pandas as pd

#postive_300 = pd.read_pickle(r'300_pos_exs.pkl')
#postive_300 = pd.read_pickle(r'63_1a2o_neg_exs.pkl')
#postive_300 = pd.read_pickle(r'1000_decoy_exs.pkl')
#postive_300 = pd.read_pickle(r'1000_pos_exs.pkl')
#print("postive samples:", len(postive_300))


def reformat_image(ex):
    ex[ex == 0] = 20
    ex = (ex - 2)/(20-2)
    ex = 1.0 - ex
    return ex


def fill_diagonal_distance_map(dist):
    new_list = []
    channel_number = dist.shape[0]
    for j in range(channel_number):
        Ndist = dist[j:j+1, :, :]
        Ndist = np.squeeze(Ndist)
        np.fill_diagonal(Ndist, 1)
        new_list.append(Ndist)
    return np.array(new_list)


def check_one_distance_map(dist, k):
    Ndist = dist[k:k+1, :, :]
    Ndist = np.squeeze(Ndist)
    Ndist = pd.DataFrame(Ndist)
    print(Ndist)

def run(workdir, out_path, in_file, isDecoy = False):
    '''
    TO DO: The bad_sample must be related to some bugs. Need to be fixed.
    '''
    os.makedirs(workdir + out_path, exist_ok=True)

    postive_300 = pd.read_pickle(workdir + in_file)

    # bad_sample =[126,291,343,345,346,373,383,385,398,580,600,625,
    #             672,793,984]
    bads = []

    tag = '_1.npy'
    if isDecoy:
        tag = '_0.npy'

    for i in range(len(postive_300)):
        one_protein = postive_300[i]
        print("protein shape:", one_protein.shape)
        
        # if i in bad_sample:
        #     continue
        try:
            dist = one_protein[0:4, :, :]    
            dist_new = reformat_image(dist.copy())
            dist_new = fill_diagonal_distance_map(dist_new)
                
            print("min, max", dist_new.min(), dist_new.max())
            print("dist_new:", dist_new.shape)
            check_one_distance_map(dist_new, 3)
            
            # combine distance and residue information
            dist_new = np.concatenate([dist_new, one_protein[4:, :, :]], axis = 0)
            print("final dist_new:", dist_new.shape)
            check_one_distance_map(dist_new, 13)
                                    
            protein_name = "bindcore_" + str(i)
            np.save(workdir + out_path + protein_name + tag, dist_new)

        except:
            bads.append(i)

    #Print out the bads for debug purpose
    print('Bads: {}'.format(bads))
    return

def npy2csv(workdir, in_path, csv_name = 'samples_both.csv'):

    with open(workdir + in_path + csv_name, 'w') as f:
        for file in os.listdir(workdir + in_path):
            if '.npy' in file:
                f.write(file + '\n')
    return 



