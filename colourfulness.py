# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:55:16 2021

@author: Yucheng
"""


def colourfulnessMetric(img):
    """
    

    Parameters
    ----------
    img : RGB image

    Returns
    -------
    M : colourness metric
    -----------------------------
    |not colourful        | 0   |
    |slightly colorful    | 15  |
    |moderately colourful | 33  | 
    |averagely colourful  | 45  | 
    |quite colourful      | 59  |
    |highly colourful     | 82  |
    |extremely colourful  | 109 |
    -----------------------------
    
    """
    # Get RGB components
    R,G,B = cv2.split(img.astype("float"))
    
    # colourfulness metric from Hasler et al., section 7
    rg = R - G
    yb = (1/2) * (R+G) - B
    
    sigma_rgyb = np.sqrt(np.var(rg) + np.var(yb))
    mu_rgyb = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    
    M = sigma_rgyb + 0.3 * mu_rgyb
    
    return M

from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import cv2
import numpy as np

img2 = cv2.imread("rainbow.jpg") # opens as BGR
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


plt.imshow(img2[:,:,:])
plt.show()

M = colourfulnessMetric(img2)
print(M)

    
        