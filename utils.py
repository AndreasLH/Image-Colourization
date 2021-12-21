import os
from PIL import Image
from skimage.color import rgb2hsv
import glob
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import torch
from torch import nn
import time
from matplotlib import pyplot as plt

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


def create_dataset():
    '''sets up folder structure for data set'''
    if not os.path.isdir("data/val_256_NOBW"):
        path = os.getcwd()
        paths = glob.glob(path + "/data/val_256/*.jpg")
        os.mkdir("data/val_256_NOBW")

        def remove_BW_images(path):
            '''removes BW images from dataset original dataset'''
            new_paths = []
            for img_path in path:
                im = Image.open(img_path)
                imt = np.array(im)
                if len(imt.shape) == 3:
                    temp = rgb2hsv(imt)
                    if temp[0,0,0] != 0:
                        new_paths.append(img_path)
                        im.save(img_path.split('/')[0]+os.sep+'data'+os.sep+'val_256_NOBW'+os.sep+img_path[-26:])
            return new_paths

        new_train_paths = remove_BW_images(paths)
        return new_train_paths


SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', mode='standard'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip() # A little data augmentation!
            ])
            # Image.BICUBIC has changed to 
            # transforms.InterpolationMode.BICUBIC
            # in newer pytorch
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        
        self.split = split
        self.size = SIZE
        self.paths = paths
        self.mode = mode
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype("float32") 
        img_lab = transforms.ToTensor()(img_lab)
        L = 2*(img_lab[[0], ...] / 255.0) -1 # normalise to [-1,1] # *
        ab = 2*(img_lab[[1, 2], ...] / 255.0) -1 # normalise to [-1,1] # *

        
        if self.mode == 'xception':
            L_temp = L
            L = torch.cat((L_temp,L_temp,L_temp),0)

        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=16, n_workers=0, pin_memory=False, **kwargs): # 
    """
    A handy function to make our dataloaders
    select mode: mode indicates which backbone you are using for the encoder.
    Options are:

    * 'standard': used for the baseline Unet
    * 'vgg19', used for the vgg19 backbone
    * 'xception', used for the xception backbone. 
    
    input these as strings with lowercase letters
    """
    if cuda:
        pin_memory = True
        n_workers = 2 ## 2 seems to be the most efficient for colab
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
    return dataloader


def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab): 
    """
    Takes a batch of images
    """
    L = (L[:,0,:,:].view(-1,1,256,256)+1.)/2 * 255.0 # Back to range [0, 255]
    ab = (ab+1.)/2 * 255.0 # Back to range [0,255]
    L = L.type(torch.uint8)
    ab = ab.type(torch.uint8)

    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = cv2.cvtColor(img,cv2.COLOR_LAB2RGB) #
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def lab_to_BW(L):
    return L[0, 0, :, :]
    
def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")



if __name__ == 'main':
    create_dataset()
