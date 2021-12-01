#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import torch

import glob
import os

from PIL import Image
from torch import nn, optim
from tqdm.notebook import tqdm
from torchvision.models.vgg import vgg19

# import sys # might be necessary
# sys.path.append('..')
from utils import  make_dataloaders, AverageMeter, create_loss_meters, update_losses, log_results
from nets import  MainModel, build_timm_unet, build_vgg_unet

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print("training using:", device)


path = os.getcwd()
paths = glob.glob(path +os.sep+"val_256_NOBW"+os.sep+"*.jpg")
# Grabbing all the image file names
np.random.seed(123)
N = len(paths) # 32509
paths_subset = np.random.choice(paths, N, replace=False) # choosing 1000 images randomly
rand_idxs = np.random.permutation(N)
train_idxs = rand_idxs[:30000] # choosing the first 30000 as training set
val_idxs = rand_idxs[30000:] # choosing the remaining as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))


mode = 'xception'
SIZE = 256
batch_size = 4

#select mode: mode indicates which backbone you are using for the encoder.
#Options are:
#'standard': used for the baseline Unet
#'vgg19', used for the vgg19 backbone
#'xception, used for the xception backbone. 
#input these as strings with lowercase letters

train_dl = make_dataloaders(paths=train_paths, batch_size=batch_size, split='train', mode=mode)
val_dl = make_dataloaders(paths=val_paths, batch_size=batch_size, split='val',mode=mode)

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))

# In[]
def pretrain_generator(net_G, train_dl, opt, criterion, epochs, mode):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")
        
save_file_name = '{}.pt'.format(mode) #name of the file where you save your weights.
if mode == 'xception':
    net_G = build_timm_unet("xception")
elif mode == 'vgg19':
    net_G = build_vgg_unet(vgg19)
opt = optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()
pretrain_generator(net_G, train_dl, opt, criterion, 20, mode)
torch.save(net_G.state_dict(), save_file_name)

# In[ ]:


def train_model(model, train_dl, epochs, display_every=100):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    optim_G = model.model_optimizers()['G']
    optim_D = model.model_optimizers()['D']

    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network'

        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
               
        if e % 10 == 9:
            print("Saving model.")
            torch.save({"epoch": e+1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict_G": optim_G.state_dict(),
                        "optimizer_state_dict_D": optim_D.state_dict()},
                        "backbone_vgg_epoch_{}.pt".format(e+1))

# In[ ]:

if mode =='vgg19':
    net_G = build_vgg_unet(vgg19)
elif mode=='xception':
    net_G = build_timm_unet('xception')
else:
    net_G = None

# use pretrained backbone if it exists.
if os.path.isfile("{}.pt".format(mode)): 
    net_G.load_state_dict(torch.load(save_file_name, map_location=device))
model = MainModel(net_G=net_G)
train_model(model, train_dl, 100)
