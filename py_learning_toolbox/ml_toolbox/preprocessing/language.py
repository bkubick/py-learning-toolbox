# coding: utf-8

from __future__ import annotations

import re
import string
import typing

import tensorflow as tf


__all__ = [
    'clean_text',
    'create_sequences_from_sentences',
    'create_sequenced_data_labels_from_sentences',
]


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


def create_sequences_from_sentences(sentences: typing.List[str]) -> tf.Tensor:
    """ Creates sequences of sentences from the given sentences.

        Example:
            Input: ['I love TensorFlow and AI']
            Output: ['I love', 'I love TensorFlow', 'I love TensorFlow and', 'I love TensorFlow and AI']

        Args:
            sentences (List[str]): The sentences to create sequences from.

        Returns:
            (tf.Tensor) The sequences of sentences.
    """
    sequences = []
    for sentence in sentences:
        words_in_sentence = sentence.split(' ')
        for i in range(2, len(words_in_sentence) + 1):
            sequences.append(' '.join(words_in_sentence[:i]))

    return tf.convert_to_tensor(sequences, dtype=tf.string)


def create_sequenced_data_labels_from_sentences(sentences: typing.List[str]) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """ Generates sequenced data from the given sentences and returns the data and labels.

        Example:
            Input: ['I love TensorFlow and AI']
            Output: (['I love', 'I love TensorFlow', 'I love TensorFlow and'],
                    ['TensorFlow', 'and', 'AI'])

        Args:
            sentences (List[str]): The sentences to create sequences from.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor]) The sequenced data.
    """
    data, labels = [], []
    for sentence in sentences:
        words_in_sentence = sentence.split(' ')
        for i in range(1, len(words_in_sentence)):
            data.append(' '.join(words_in_sentence[:i]))
            labels.append(words_in_sentence[i])

    return tf.convert_to_tensor(data, dtype=tf.string), tf.convert_to_tensor(labels, dtype=tf.string)
