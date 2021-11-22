# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:21:38 2021

@author: Yucheng
"""

import numpy as np

a = np.arange(0,256)
b = 2*(a / 255.0) - 1

c = (b+1)/2* 255

print(a)
print(b)
print(c)

#%%
import cv2 
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("mesprit.jpg").convert("RGB")
img = np.array(img)
plt.imshow(img)
plt.title("1")
plt.show()
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

L = 2*(img_lab[...,[0]] / 255.0) -1 # normalise to [-1,1] # *
ab = 2*(img_lab[...,[1,2]] / 255.0) -1 # normalise to [-1,1] # *

L2 = (L+1)/2 *255.
ab2 = (ab+1)/2 * 255.
L2 = L2.astype("uint8")
ab2 = ab2.astype("uint8")

img_recreated = np.concatenate((L2,ab2),axis=2)



plt.imshow(img_recreated)
plt.title("2")
plt.show()
img2 = cv2.cvtColor(img_recreated, cv2.COLOR_LAB2RGB)
plt.imshow(img2)
plt.title("3")
plt.show()

#%%
import torch


g = torch.Tensor(3,2)