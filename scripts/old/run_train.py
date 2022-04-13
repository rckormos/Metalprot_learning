import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'src'))
from trainer import train


# PROJ_PATH = '/home/ybwu/projects/Protein/testing/mbc/'
# DATA_PATH = PROJ_PATH + 'input/'
# out_dir = '/home/ybwu/projects/Protein/testing/mbc/Aug1221_AF1/'

#BATCH_SIZE = 16
#EPOCHS = 20
 
workdir = os.getcwd()
DATA_PATH = workdir + '/data/input/out/'
out_dir = DATA_PATH + 'classify_results/'

os.makedirs(out_dir, exist_ok=True)

train.run(DATA_PATH, out_dir, BATCH_SIZE = 16, EPOCHS = 20)

