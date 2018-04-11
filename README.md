# pytorch-msssim

### Differentiable Multi-Scale Structural Similarity (SSIM) index 

This small utiliy provides a differentiable MS-SSIM implementation for PyTorch based on Po Hsun Su's implementation of SSIM @ https://github.com/Po-Hsun-Su/pytorch-ssim.
At the moment only a direct method is supported.

## Installation

1. Go to the repo directory.
2. Run `python setup.py install`

or 

1. Clone this repo.
2. Copy "pytorch_msssim" folder in your project.


## Example
The provided images for calculation must be Variables.

### basic usage
```python
import pytorch_ssim
import torch
from torch.autograd import Variable

m = pytorch_msssim.MSSSIM()

img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))


if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print(pytorch_msssim.msssim(img1, img2))
print(m(img1, img2))


```

## Reference
https://ece.uwaterloo.ca/~z70wang/research/ssim/
https://github.com/Po-Hsun-Su/pytorch-ssim
