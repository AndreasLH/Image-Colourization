#!/bin/sh
#BSUB -J backbone
#BSUB -o backbone%J.out
#BSUB -e backbone%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=8G]"
#BSUB -W 4:00
#BSUB -N
# end of BSUB options

module load python3/3.9.6

# load CUDA (for GPU support)
module load cuda/11.3

# activate the virtual environment
source DeepLearning/bin/activate

python backbone_hpc.py
