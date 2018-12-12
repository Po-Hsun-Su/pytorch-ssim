# pytorch-msssim

### Differentiable Multi-Scale Structural Similarity (SSIM) index 

This small utiliy provides a differentiable MS-SSIM implementation for PyTorch based on Po Hsun Su's implementation of SSIM @ https://github.com/Po-Hsun-Su/pytorch-ssim.
At the moment only a direct method is supported.

## Installation

Master branch now only supports PyTorch 0.4 or higher. All development occurs in the dev branch (`git checkout dev` after cloning the repository to get the latest development version).

To install the current version of pytorch_mssim:

1. Clone this repo.
2. Go to the repo directory.
3. Run `python setup.py install`

or 

1. Clone this repo.
2. Copy "pytorch_msssim" folder in your project.

To install a version of of pytorch_mssim that runs in PyTorch 0.3.1 or lower use the tag checkpoint-0.3. To do so, run the following commands after cloning the repository:

```
git fetch --all --tags
git checkout tags/checkpoint-0.3
```

## Example

### Basic usage
```python
import pytorch_msssim
import torch
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = pytorch_msssim.MSSSIM()

img1 = torch.rand(1, 1, 256, 256)
img2 = torch.rand(1, 1, 256, 256)

print(pytorch_msssim.msssim(img1, img2))
print(m(img1, img2))


```

### Training

For a detailed example on how to use msssim for training, look at the file max_ssim.py.

We recommend using the flag normalized=True when training unstable models using MS-SSIM (for example, Generative Adversarial Networks) as it will guarantee that at the start of the training procedure, the MS-SSIM will not provide NaN results.

## Reference
https://ece.uwaterloo.ca/~z70wang/research/ssim/

https://github.com/Po-Hsun-Su/pytorch-ssim

Thanks to z70wang for providing the initial SSIM implementation and all the contributors with fixes to this fork.
