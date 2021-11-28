#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
### -- set the job Name --
#BSUB -J GAN_backbone_V
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s194241@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# Commands
# linuxsh / sxm2sh / voltash
# bsub < jobscript.sh
# bstat
# bkill

#nvidia-smi
#lscpu
#module load cudnn

#module avail > modules.txt

#comment 
#from PyTorch.SpeechSynthesis.Tacotron2.models import batchnorm_to_float
#in file
#/zhome/43/7/137749/.cache/torch/hub/nvidia_DeepLearningExamples_torchhub/hubconf.py


module load python3/3.9.6

# load CUDA (for GPU support)
module load cuda/11.3

# activate the virtual environment
source DeepLearning/bin/activate

python backbone_hpc.py

