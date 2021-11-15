from math import log10, sqrt
import cv2
import numpy as np

def PSNR(original, compressed):
    '''
    Calculates the Peak signal to noise ratio between a ground truth image and predicted image.

    input: 
        true image (cv2 image)
        predicted image (cv2 image)
    output:
        PSNR score
    '''
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
def main():
     original = cv2.imread("test_imgs/original_image.png")
     compressed = cv2.imread("test_imgs/compressed_image1.png", 1)
     value = PSNR(original, compressed)
     print(f"PSNR value is {value} dB")
       
if __name__ == "__main__":
    main()
