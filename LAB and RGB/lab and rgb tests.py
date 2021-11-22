# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:07:01 2021

@author: Yucheng
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2hsv, rgb2lab
from PIL import Image
from copy import copy

img = cv2.imread("shariatna.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = np.array(img)
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
# img_lab = rgb2lab(img)



cv2.waitKey(0)
cv2.destroyAllWindows()

def plotMinMax(Xsub_rgb,labels=["R","G","B"]):
    print("______________________________")
    for i, lab in enumerate(labels):
        mi = np.min(Xsub_rgb[:,:,i])
        ma = np.max(Xsub_rgb[:,:,i])
        print("{} : MIN={:8.4f}, MAX={:8.4f}".format(lab,mi,ma))
        
plotMinMax(img,labels=["R","G","B"])    

plotMinMax(img_lab, labels=["L","A","B"])


def extract_single_dim(image, dim):
    z = np.zeros(image.shape).astype("float32")
    if dim != 0:
        z[:,:,0] = 80
    z[:,:,dim] = image[:,:,dim]
    z = cv2.cvtColor(z, cv2.COLOR_LAB2RGB)
    return z

L = extract_single_dim(img_lab,0)
a = extract_single_dim(img_lab,1)
b = extract_single_dim(img_lab,2)


fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(8, 2))

ax0.imshow(img)
ax0.set_title("RGB image")
ax0.axis("off")
ax1.imshow(L)
ax1.set_title("L channel")
ax1.axis("off")
ax2.imshow(a)
ax2.set_title("A* channel")
ax2.axis("off")
ax3.imshow(b)
ax3.set_title("B* channel")
ax3.axis("off")
plt.show()

plt.imshow(L)
plt.show()
plt.imshow(a)
plt.show()
plt.imshow(b)
plt.show()


# for i, lab in enumerate(["R", "G", "B"]):
#     crgb = np.zeros(img.shape)
#     print(crgb.shape)
#     crgb[:,:,i] = img[:,:,i]
#     plt.imshow(crgb / 255.0)
#     plt.show()

    

plotMinMax(img_lab, labels=["L","A","B"])

def extract_single_dim(image, idim):
    z = np.zeros(image.shape)
    if idim != 0:
        z[:,:,0] = 80
    z[:,:,idim] = image[:,:,idim]
    z = lab2rgb(z)
    
    return z


lab_rgb_L = extract_single_dim(img_lab, 0)
plt.imshow(lab_rgb_L)
plt.show()

lab_rgb_a = extract_single_dim(img_lab, 1)
plt.imshow(lab_rgb_a)
plt.show()

lab_rgb_b = extract_single_dim(img_lab, 2)
plt.imshow(lab_rgb_b)
plt.show()