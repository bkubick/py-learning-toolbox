#!/usr/bin/env python3
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name='py-learning-toolbox',
    version='0.0.0-dev',
    description='Python toolbox for my perosnal AI, machine learning, genetic algorithms, and deep learning projects.',
    author='Brandon Kubick',
    packages=find_packages(),
    include_package_data = True,
    package_data={'py_learning_toolbox': ['_commands/templates/notebook.json']},
    install_requires=[
        'matplotlib>=3.7.1',
        'numpy>=1.23.5',
        'pandas>=1.5.3',
        'scikit-learn>=1.2.2',
        'tensorboard>=2.12.3',
        'tensorflow>=2.12.0',
        'tensorflow-hub>=0.14.0',
        'tensorflow-model-optimization>=0.6.0',
    ],
    entry_points={
        'console_scripts': [
            'pltb = py_learning_toolbox._commands.main:main', 
        ],
    },
)
