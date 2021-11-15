import numpy as np
  
def PSNR(true, pred):
    '''
    Calculates the Peak signal to noise ratio between a ground truth image and predicted image.

    input: 
        true image (PIL Image in RGB format)
        predicted image (PIL Image in RGB format)
    output:
        PSNR score
    '''
    true = np.array(true)
    pred = np.array(pred)
    mse = np.mean((true - pred) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
  
  
def main():
    from PIL import Image
    original = Image.open("test_imgs/original_image.png")
    compressed = Image.open("test_imgs/compressed_image1.png")
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")
       
if __name__ == "__main__":
    main()