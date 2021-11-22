from math import log10, sqrt
import cv2
import numpy as np

def PSNR(original, compressed):
    '''
    Calculates the Peak signal to noise ratio between a ground truth image and predicted image.

    Parameters
    ----------
        true image (cv2 image)
        predicted image (cv2 image)
        
    Returns
    -------
        PSNR score
    '''
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:55:16 2021

@author: Yucheng
"""

import cv2
import numpy as np



def colourfulnessMetric(img):
    """
    Created on Mon Nov 15 10:55:16 2021

    @author: Yucheng

    Parameters
    ----------
    img : cv2 RGB image

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

def main():
    import matplotlib.pyplot as plt
    original = cv2.imread("test_imgs/original_image.png")
    compressed = cv2.imread("test_imgs/compressed_image1.png", 1)
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")


    img2 = cv2.imread("rainbow.jpg") # opens as BGR
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.imshow(img2[:,:,:])
    plt.show()

    M = colourfulnessMetric(img2)
    print(M)
       
if __name__ == "__main__":
    main()
