# coding: utf-8

from __future__ import annotations

import re
import string


__all__ = ['clean_text']


def clean_text(text: str,
                    strip_punctuation: bool = True,
                    lowercase: bool = True,
                    strip_leading_and_trailing_whitespace: bool = True,
                    strip_spacing_characters: bool = True) -> str:
    """ Cleans the given text.

        Args:
            text (str): The text to preprocess.
            strip_punctuation (bool): Whether to remove punctuation from the text.
            lowercase (bool): Whether to lowercase the text.
            strip_leading_and_trailing_whitespace (bool): Whether to remove whitespace from the text.
            strip_spacing_characters (bool): Whether to remove spacing characters from the text.

        Returns:
            (str) The preprocessed text.
    """
    text = text.lower()

    if strip_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if lowercase:
        text = text.lower()
    
    if strip_leading_and_trailing_whitespace:
        text = text.strip()
    
    if strip_spacing_characters:
        text = re.sub(r'\s+', ' ', text)

    return text
