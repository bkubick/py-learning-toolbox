# coding: utf-8
""" This package contains all the utilities for working with data. This includes functions for loading and
    structuring data.
"""

from .utils import read_txt_file_from_directory, read_txt_file_from_url


__all__ = [
    'read_txt_file_from_directory',
    'read_txt_file_from_url',
]
