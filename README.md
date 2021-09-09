# pytorch-ssim (This repo is not maintained) 

The code doesn't work because it is on super old pytorch.

### Differentiable structural similarity (SSIM) index.
![einstein](https://raw.githubusercontent.com/Po-Hsun-Su/pytorch-ssim/master/einstein.png) ![Max_ssim](https://raw.githubusercontent.com/Po-Hsun-Su/pytorch-ssim/master/max_ssim.gif)

## Installation
1. Clone this repo.
2. Copy "pytorch_ssim" folder in your project.

## Example
### basic usage
```python
import pytorch_ssim
import torch
from torch.autograd import Variable

img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print(pytorch_ssim.ssim(img1, img2))

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

print(ssim_loss(img1, img2))

```
### maximize ssim
```python
import pytorch_ssim
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


# Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
ssim_value = pytorch_ssim.ssim(img1, img2).data[0]
print("Initial ssim:", ssim_value)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
ssim_loss = pytorch_ssim.SSIM()

optimizer = optim.Adam([img2], lr=0.01)

while ssim_value < 0.95:
    optimizer.zero_grad()
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = - ssim_out.data[0]
    print(ssim_value)
    ssim_out.backward()
    optimizer.step()

```

## Reference
https://ece.uwaterloo.ca/~z70wang/research/ssim/
