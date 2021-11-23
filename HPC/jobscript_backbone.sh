#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
### -- set the job Name --
#BSUB -J GAN_backbone
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


cd /zhome/bb/d/146781/Desktop/Deep/

module load python3/3.8.0
module load cuda/11.0




pip3 install --user --upgrade pip
pip3 install --user unidecode
pip3 install --user inflect
pip3 install --user matplotlib
pip3 install --user numpy
pip3 install --user torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user glob2
pip3 install --user Pillow
pip3 install --user torchvision
pip3 install --user tqdm
pip3 install --user opencv-python
pip3 install --user fastai --upgrade
pip3 install --user timm

#Generate wav files
python3 backbone_hpc.py



