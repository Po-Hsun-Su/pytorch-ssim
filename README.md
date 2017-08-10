# pytorch-ssim

#### Differentiable structure similarity (SSIM) index.
![einstein](https://raw.githubusercontent.com/Po-Hsun-Su/pytorch-ssim/master/einstein.png)

![Max_ssim](https://raw.githubusercontent.com/Po-Hsun-Su/pytorch-ssim/master/max_ssim.gif)

## Example

```python
import pytorch_ssim
import torch
from torch.autograd import Variable

img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))

print(pytorch_ssim.ssim(img1, img2))

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

print(ssim_loss(img1, img2))

```

## Reference
