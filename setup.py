#!/usr/bin/env python3
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name='py-learning-toolbox',
    version='0.0.2',
    description='Python toolbox for my perosnal AI, machine learning, genetic algorithms, and deep learning projects.',
    author='Brandon Kubick',
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.7.2',
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'tensorboard>=2.13.0',
        'tensorflow>=2.13.0',
        'tensorflow-hub>=0.14.0',
        'scikit-learn>=1.3.0',
    ],
    entry_points={
        'console_scripts': [
            'pltb_setup_project = py_learning_toolbox.setup_project:main',
        ],
    },
)
