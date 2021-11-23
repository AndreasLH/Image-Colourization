#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import torch

import glob
import os
import time

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Places365
from tqdm.notebook import tqdm
import cv2
from fastai.vision.learner import create_body
from torchvision.models.vgg import vgg19
from fastai.vision.models.unet import DynamicUnet
from timm import create_model

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print("training using:", device)


# In[ ]:


path = os.getcwd()
paths = glob.glob(path +os.sep+"val_256_NOBW"+os.sep+"*.jpg")
# Grabbing all the image file names
np.random.seed(123)
paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 1000 images randomly
rand_idxs = np.random.permutation(10_000)
train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
val_idxs = rand_idxs[8000:] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")


# In[ ]:


mode = 'xception'
if mode == 'xception':
    SIZE = 299
else:
    SIZE = 256
batch_size = 4
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train',mode='standard'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip() # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        
        self.split = split
        self.size = SIZE
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        # img = cv2.imread(self.paths[idx]) # *
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype("float32") # *
        # img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        # print("img shape:", img_lab.shape)
        L = 2*(img_lab[[0], ...] / 255.0) -1 # normalise to [-1,1] # *
        ab = 2*(img_lab[[1, 2], ...] / 255.0) -1 # normalise to [-1,1] # *
        # print("L shape:", L.shape)
        # print("ab shape:",ab.shape)
        # L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1 ### L channel is in range [0, 100]
        # ab = img_lab[[1, 2], ...] / 128. # Between -1 and 1 ### I think this is wrong. According to [1]: a,b channel is in range [-110,110] 
                                                            ### However, according to [2], [3] a,b channel is in range [-128,127]
        if mode == 'xception':
          L_temp = L
          L = torch.cat((L_temp,L_temp,L_temp),0)

# [1] http://ai.stanford.edu/~ruzon/software/rgblab.html
# [2] https://stackoverflow.com/questions/46415948/converting-rgb-images-to-lab-using-scikit-image
# [3] https://www.colourphil.co.uk/lab_lch_colour_space.shtml
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=16, n_workers=0, pin_memory=False, **kwargs): # A handy function to make our dataloaders
    if cuda:
        pin_memory = True
#         n_workers = 2 ## seems to be the most efficient for colab
        n_workers = 0 # for jupyter
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
    return dataloader


# In[ ]:


#select mode: mode indicates which backbone you are using for the encoder.
#Options are:
#'standard': used for the baseline Unet
#'vgg19', used for the vgg19 backbone
#'xception, used for the xception backbone. 
#input these as strings with lowercase letters

train_dl = make_dataloaders(paths=train_paths, batch_size=batch_size, split='train',mode=mode)
val_dl = make_dataloaders(paths=val_paths, batch_size=batch_size, split='val',mode=mode)

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))


# In[ ]:


class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)


# In[ ]:


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L[:,0,:,:].view(batch_size,1,SIZE,SIZE), self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L[:,0,:,:].view(batch_size,1,SIZE,SIZE), self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L[:,0,:,:].view(batch_size,1,SIZE,SIZE), self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

    def model_optimizers(self):
        opt = {'G': self.opt_G, 'D': self.opt_D}
        return opt


# In[ ]:


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

def lab_to_rgb(L, ab): ########## klamt!!
    """
    Takes a batch of images
    """
    L = (L[:,0,:,:].view(batch_size,1,SIZE,SIZE)+1.)/2 * 255.0 # Back to range [0, 255]
    ab = (ab+1.)/2 * 255.0 # Back to range [0,255]
    L = L.type(torch.uint8)
    ab = ab.type(torch.uint8)
    # L = (L + 1.) * 50.
    # ab = ab * 128.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = cv2.cvtColor(img,cv2.COLOR_LAB2RGB) # *
        # img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
    
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

def get_G_and_D_loss(loss_meter_dict, model):
    loss_G = loss_meter_dict['loss_G'].avg
    loss_D = loss_meter_dict['loss_D'].avg
    return loss_G, loss_D
  

def plot_losses(G_loss_array, D_loss_array):
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8,2))
    ax0.plot(G_loss_array)
    ax0.set_title("Generator loss")
    ax0.axis("off")
    ax1.plot(D_loss_array)
    ax1.set_title("Discriminator loss")
    ax1.axis("off")
    plt.show()


# In[ ]:


def create_timm_body(arch:str, n_in=3, pretrained=True, cut=None):
    "Cut off the body of a typically pretrained `arch` as determined by `cut`"
    model = create_model(arch,pretrained=pretrained)
    # _update_first_layer(model, n_in, pretrained)
    #cut = ifnone(cut, cnn_config(arch)['cut'])
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if   isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or a function")

def build_vgg_unet(arch, n_input=1, n_output=2, size=256):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  body = create_body(arch=arch, pretrained=True, n_in=1, cut=-2)
  net_G = DynamicUnet(body, n_output, (size, size)).to(device)
  return net_G

def build_timm_unet(arch, n_input=1, n_output=2, size=256):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  body = create_timm_body(arch=arch, pretrained=True, n_in=1, cut=-10)
  net_G = DynamicUnet(body, n_output, (size, size)).to(device)
  return net_G


# In[ ]:


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
                visualize(model, data, save=False) # function displaying the model's outputs

        if e % 10 == 0:
            print("Saving model.")
            torch.save({"epoch": e,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict_G": optim_G.state_dict(),
                        "optimizer_state_dict_D": optim_D.state_dict()},
                        "backbone_model_epoch_{}".format(e))


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

