import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def do_conv(conv_func, img, windows, channel):

    ndims = len(windows)
        
    for i in range(0, ndims):
        
        window = windows[i]
        
        padding_amt = int((np.max(window.size())-1) /2 )
        padding = [0]*(ndims)
        padding[i] = padding_amt
        padding = tuple(padding)
    
        img = conv_func(img, window, padding=padding, groups = channel)

    return img
                        
def create_windows(img, window_sizes, sigma):
    
    windows = list()
    
    ndims = len(img.size())-2
    
    if type(window_sizes) is not list:
        window_sizes = [window_sizes]*ndims
    
    for i in range(0, ndims):
        g = gaussian(window_sizes[i], 1.5)
        
        for j in range(0, ndims+1):
            g = g.unsqueeze(-1)
            
        g = g.transpose(0, i+2)
        
        g = Variable(g)
        if img.is_cuda:
            g = g.cuda(img.get_device())
        g = g.type_as(img)
            
        windows.append(g)
        
    return windows

    
def _ssim(img1, img2, windows, channel, size_average = True):
        
    ndims = len(windows)
    
    if ndims == 1:
        conv_func = F.conv1d
    if ndims == 2:
        conv_func = F.conv2d
    if ndims == 3:
        conv_func = F.conv3d
                
    mu1 = do_conv(conv_func, img1, windows, channel = channel)
    mu2 = do_conv(conv_func, img2, windows, channel = channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = do_conv(conv_func, img1*img1, windows, channel = channel) - mu1_sq
    sigma2_sq = do_conv(conv_func, img2*img2, windows, channel = channel) - mu2_sq
    sigma12 = do_conv(conv_func, img1*img2, windows, channel = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, sigma = 0.15, size_average = True):
        super(SSIM, self).__init__()
        
        self.window_size = window_size
        self.sigma = sigma
        
        self.size_average = size_average
        self.channel = 1
        
        self.windows = None

    def forward(self, img1, img2):
        imsize = img1.size()
        channel = imsize[1]

        if self.windows is None:
            self.windows = create_windows(img1, self.window_size, self.sigma)
            self.channel = channel


        return _ssim(img1, img2, self.windows, self.channel, self.size_average)

def ssim(img1, img2, window_size = 11, sigma = 1.5, size_average = True):
    imsize = img1.size()
    channel = imsize[1]
        
    windows = create_windows(img1, window_size, sigma)    
    
    return _ssim(img1, img2, windows, channel, size_average)
