# Image-Colourization
Image colourization 🖍🎨 for our project in the DTU Deep Learning course (02456)

# Milestones
- [x] Data: use the places365 dataset (remove BW images)
- [x] Make the baseline (GAN and L1-loss without transfer learning)
- [x] Test difference between L1 and L2 loss on baseline model
- [ ] Use image labels as additional conditional data and assess improvement
- [ ] Evaluate how image label data improved the model
- [x] 2 backbones VGG19, Xception
- [x] Quantitative evaluation (colourfulness, peak signal-to-noise ratio (PSNR))
- [x] Qualitative human evaluation (by us) on 5 images each (discussion in report)
- [ ] Final run with best model compared to baseline

# How to activate on HPC
$ are terminal commands
1. open terminal in same folder as this project and type the following commands (you can paste them into the terminal with middle mouse click)
2. ```$ module load python3/3.9.6```
3. ```$ module load cuda/11.3```
4. ```$ python3 -m venv DeepLearning```
5. ```$ source DeepLearning/bin/activate```
6. ```$ pip3 install -r requirements.txt```

Now everything should be setup. Then see the ```submit.sh``` shell script for how it is activated. It should be run from the same path as the project.
