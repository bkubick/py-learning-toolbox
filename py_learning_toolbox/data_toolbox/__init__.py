# coding: utf-8
""" This package contains all the utilities for working with data. This includes functions for loading and
    structuring data into datasets.
"""

from .datasets import generate_dataset_from_data, split_data_labels_from_dataset
from .reading import read_txt_file_from_directory, read_txt_file_from_url


__all__ = [
    'generate_dataset_from_data',
    'read_txt_file_from_directory',
    'read_txt_file_from_url',
    'split_data_labels_from_dataset',
]
