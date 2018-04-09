from distutils.core import setup

setup(
  name = 'pytorch_msssim',
  packages = ['pytorch_msssim'], # this must be the same as the name above
  version = '0.1',
  description = 'Differentiable multi-scale structural similarity (MS-SSIM) index',
  author = 'Jorge Pessoa',
  author_email = 'jpessoa.on@gmail.com',
  url = 'https://github.com/jorge-pessoa/pytorch-msssim', # use the URL to the github repo
  keywords = ['pytorch', 'image-processing', 'deep-learning', 'ms-ssim'], # arbitrary keywords
  classifiers = [],
)
