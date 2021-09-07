# Reload module
In the development process, you can use importlib for the reload purpose.

import sys 
import importlib  
importlib.reload(sys.modules['src.manipulator.dataprocess'])



# run cpu in wynton. 
qsub -cwd -l run_gpu_rnn

# run gpu in wynton. 
qsub -cwd -j yes -q gpu.q run_tran.sh 
qsub -cwd -j yes -q gpu.q -pe smp 10 run_tran.sh 

qsub -cwd -j yes -q gpu.q -l compute_cap=61 run_tran.sh 

