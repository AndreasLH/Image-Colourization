# Image Colourization with conditional GANs
Repository for our image colourization 🖍🎨 project in the Deep Learning course (02456) at DTU. 

You can find our poster [here](poster/02456_Deep_Learning_Image_Colourization.pdf).

# How it works

# Milestones
- [x] Data: use the places365 dataset (remove BW images)
- [x] Make the baseline (GAN and L1-loss without transfer learning)
- [x] Test difference between L1 and L2 loss on baseline model
- [x] 2 backbones VGG19, Xception
- [x] Quantitative evaluation (colourfulness, peak signal-to-noise ratio (PSNR))
- [x] Qualitative human evaluation (by us) on 5 images each (discussion in report)
- [ ] Use image labels as additional conditional data and assess improvement
- [ ] Evaluate how image label data improved the model

# Training your own model on HPC cluster
The Places365 dataset is available [here](http://places2.csail.mit.edu/): 

$ are terminal commands
1. open terminal in same folder as this project and type the following commands (you can paste them into the terminal with middle mouse click)
2. ```$ module load python3/3.9.6```
3. ```$ module load cuda/11.3```
4. ```$ python3 -m venv DeepLearning```
5. ```$ source DeepLearning/bin/activate```
6. ```$ pip3 install -r requirements.txt```

Now everything should be setup. Then see the ```HPC/submit.sh``` shell script for how it is activated. It should be run from the same path as the project.
