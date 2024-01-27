from setuptools import setup, find_packages

setup(
name="FashionNeRF",
description="A simple package.",
version="1.0.0",
author="Name",
author_email="Name@domain.com",
packages=find_packages(exclude=[
        'preprocessing',
        'scripts',
        'runs',
        'slurms',
        'wandb',
        'results',
        'tensorboard',
        'experiments',
        'checkpoints',
    ])
)