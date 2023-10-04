# coding: utf-8

from __future__ import annotations

import re
import string
import typing


__all__ = [
    'clean_text',
    'create_sequences_from_sentences',
    'create_sequenced_data_labels_from_sentences',
]

if typing.TYPE_CHECKING:
    DataLabelTuple = typing.Tuple[typing.List[str], typing.List[str]]


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


def create_sequences_from_sentences(sentences: typing.List[str],
                                    level: typing.Optional[str] = None) -> typing.List[str]:
    """ Creates sequences of sentences from the given sentences.

        Example:
            Input: ['I love TensorFlow and AI']
            Output: ['I love', 'I love TensorFlow', 'I love TensorFlow and', 'I love TensorFlow and AI']

        Raises:
            ValueError: If the level is not one of ['word', 'character'].

        Args:
            sentences (List[str]): The sentences to create sequences from.
            level (Optional[str]): The level to create sequences from.
                Must be one of ['word', 'character'], defaults to 'word'.

        Returns:
            (typing.List[str]) The sequences of sentences.
    """
    level = level or 'word'

    sequences = []
    for sentence in sentences:
        if level == 'word':
            words_in_sentence = sentence.split(' ')
            for i in range(2, len(words_in_sentence) + 1):
                sequences.append(' '.join(words_in_sentence[:i]))
        elif level == 'character':
            for i in range(2, len(sentence) + 1):
                sequences.append(sentence[:i])
        else:
            raise ValueError(f'level must be one of ["word", "character"], got {level}')

    return sequences


def create_sequenced_data_labels_from_sentences(sentences: typing.List[str],
                                                level: typing.Optional[str] = None) -> DataLabelTuple:
    """ Generates sequenced data from the given sentences and returns the data and labels.

        Example:
            Input: ['I love TensorFlow and AI']
            Output: (['I love', 'I love TensorFlow', 'I love TensorFlow and'],
                    ['TensorFlow', 'and', 'AI'])

        Raises:
            ValueError: If the level is not one of ['word', 'character'].

        Args:
            sentences (List[str]): The sentences to create sequences from.
            level (Optional[str]): The level to create sequences from.
                Must be one of ['word', 'character'], defaults to 'word'.

        Returns:
            (Tuple[typing.List[str], typing.List[str]]) The sequenced data and labels.
    """
    level = level or 'word'

    data, labels = [], []
    for sentence in sentences:
        if level == 'word':
            words_in_sentence = sentence.split(' ')
            for i in range(1, len(words_in_sentence)):
                data.append(' '.join(words_in_sentence[:i]))
                labels.append(words_in_sentence[i])
        elif level == 'character':
            for i in range(1, len(sentence)):
                data.append(sentence[:i])
                labels.append(sentence[i])
        else:
            raise ValueError(f'level must be one of ["word", "character"], got {level}')

    return data, labels
