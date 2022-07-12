import re
from setuptools import setup, find_packages

setup(
  name='experiment',
  version=open('.version').read(),
  description='A model library of tensorflow using for experiment',
  author='Terence Wang',
  author_email='wangteng18@mails.ucas.ac.cn',
  license='MIT',
  keywords='tensorflow, model zoo, experiment',
  packages=find_packages(),
  install_requires=['numpy','tensorflow'],
  python_requires='>=3.6'
)
