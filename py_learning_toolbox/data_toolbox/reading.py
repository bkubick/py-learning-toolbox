# coding: utf-8

from __future__ import annotations

import requests
import typing


__all__ = ['read_txt_file_from_directory', 'read_txt_file_from_url']


def read_txt_file_from_directory(filepath: str, delimiter: typing.Optional[str] = None) -> typing.List[str]:
    """ Reads a txt file from a filepath, and splits it by a delimiter.
    
        Args:
            filepath (str): The filepath to the txt file.
            delimiter (Optional[str]): The delimiter to split the text file by.

        Raises:
            TypeError: If the filepath does not end with .txt.

        Returns:
            (List[str]) The processed data from the txt file.
    """
    if not filepath.endswith('.txt'):
        raise TypeError('filepath must direct to a file that ends with .txt')

    if delimiter is None:
        delimiter = '\n'

    with open(filepath, 'r') as f:
        raw_data = f.read().split(delimiter)

    return raw_data


def read_txt_file_from_url(url: str, delimiter: typing.Optional[str] = None) -> typing.List[str]:
    """ Reads a txt file from a url, and splits it by a delimiter.
    
        Args:
            url (str): The url to the txt file.
            delimiter (Optional[str]): The delimiter to split the txt file by.

        Raises:
            TypeError: If the url does not end with .txt.
            ValueError: If the url is not valid.

        Returns:
            (List[str]) The processed data from the txt file.
    """
    if not url.endswith('.txt'):
        raise TypeError('Url must direct to a file that ends with .txt')

    if delimiter is None:
        delimiter = '\n'

    try:
        req = requests.get(url)
        raw_text = req.text
    except Exception:
        raise ValueError('Url is not valid')

    return raw_text.split(delimiter)
