# coding: utf-8

from __future__ import annotations

import string


__all__ = ['clean_text']


def clean_text(text: str,
                    strip_punctuation: bool = True,
                    lowercase: bool = True,
                    strip_whitespace: bool = False) -> str:
    """ Cleans the given text.

        Args:
            text (str): The text to preprocess.
            strip_punctuation (bool): Whether to remove punctuation from the text.
            lowercase (bool): Whether to lowercase the text.
            strip_whitespace (bool): Whether to remove whitespace from the text.

        Returns:
            (str) The preprocessed text.
    """
    text = text.lower()

    if strip_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if lowercase:
        text = text.lower()
    
    if strip_whitespace:
        text = text.strip()

    return text
