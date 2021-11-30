#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import torch

import glob
import os
from tqdm import tqdm

# import sys # might be necessary
# sys.path.append('..')
from utils import make_dataloaders, create_loss_meters, update_losses, log_results
from nets import  MainModel

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print("training using:", device)
# In[2]:


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

SIZE = 256

# In[4]:
train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))


def train_model(model, train_dl, epochs, display_every=100):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    optim_G = model.model_optimizers()['G']
    optim_D = model.model_optimizers()['D']

#     loss_G_array, loss_D_array = [], []
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network'
#         loss_G, loss_D = get_G_and_D_loss(loss_meter_dict, model)
#         loss_G_array.append(loss_G)
#         loss_D_array.append(loss_D)
#         plot_losses(loss_G_array, loss_D_array)
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
                        "model_L2_epoch_{}.pt".format(e+1))

model = MainModel()
train_model(model, train_dl, 100)
