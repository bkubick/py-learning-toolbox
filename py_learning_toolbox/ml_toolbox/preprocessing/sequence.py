# coding: utf-8

from __future__ import annotations

import logging
import typing

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if typing.TYPE_CHECKING:
    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = [
    'make_train_test_split',
]


def make_train_test_split(data: tf.Tensor, labels: tf.Tensor, test_split: float = 0.2) -> typing.Tuple:
    """ Splits matching pairs of data and labels into train and test splits.

        Args:
            data (tf.Tensor): the data to split.
            labels (tf.Tensor): the labels corresponding to the windowed data
            test_split (float): the fraction of data dedicated to be used as test set,
                must fall between 0-1.

        Raises:
            AssertionError: when test_split does not fall between 0-1.

        Returns
            (typing.Tuple) the train-test split of the windows and labels
                -> (X_train, X_test, y_train, y_test).
    """
    assert (test_split >= 0 and test_split <= 1), 'test_split must be between 0-1'

    split_index = int(len(data) * (1 - test_split))

    X_train = data[:split_index]
    y_train = labels[:split_index]
    X_test = data[split_index:]
    y_test = labels[split_index:]

    return X_train, X_test, y_train, y_test
