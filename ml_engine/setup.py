# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
  'pendulum==2.1.2', 
  'absl-py==0.11.0', 
  'transformers==4.5.1', 
  'datasets==1.4.1', 
  'wandb==0.10.21',
  'nltk==3.4.4', 
  'pandas==1.1.4', 
  'tabulate==0.8.7', 
  'ray>=1.0.0,<=1.2.0', 
  'modin[ray]==0.8.2', 
  'fairscale==0.3.2', 
  'pickle5==0.0.10'
]

PACKAGE_DATA = [
  'project_config/sms-augmentation_config.yaml',
  'project_config/general_config.yaml'
]

setup(name='ml-engine',
      version='1.0',
      packages=find_packages(),
      package_dir={'trainer': 'trainer'},
      package_data={'trainer': PACKAGE_DATA},
      description='SMS Augmentation',
      author='Mars.Su',
      author_email='hueiyuansu@gmail.com',
      license='MIT',
      install_requires=REQUIRED_PACKAGES,
      python_requires='>=3.8',
      zip_safe=False)
