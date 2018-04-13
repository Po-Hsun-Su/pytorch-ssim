import pytorch_msssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np

npImg1 = cv2.imread("einstein.png")

img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
img2 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2, requires_grad = True)


# Functional: pytorch_msssim.msssim(img1, img2, window_size = 11, size_average = True)
msssim_value = pytorch_msssim.msssim(img1, img2).data[0]
print("Initial msssim:", msssim_value)

# Module: pytorch_msssim.SSIM(window_size = 11, size_average = True)
msssim_loss = pytorch_msssim.SSIM()

optimizer = optim.Adam([img2], lr=0.01)

while msssim_value < 0.95:
    optimizer.zero_grad()
    msssim_out = -msssim_loss(img1, img2)
    msssim_value = - msssim_out.data[0]
    print(msssim_value)
    msssim_out.backward()
    optimizer.step()
