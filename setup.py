from distutils.core import setup
setup(
  name = 'pytorch_ssim',
  packages = ['pytorch_ssim'], # this must be the same as the name above
  version = '0.1',
  description = 'Differentiable structural similarity (SSIM) index',
  author = 'Po-Hsun (Evan) Su',
  author_email = 'evan.pohsun.su@gmail.com',
  url = 'https://github.com/Po-Hsun-Su/pytorch-ssim', # use the URL to the github repo
  download_url = 'https://github.com/Po-Hsun-Su/pytorch-ssim/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['pytorch', 'image-processing', 'deep-learning'], # arbitrary keywords
  classifiers = [],
)
