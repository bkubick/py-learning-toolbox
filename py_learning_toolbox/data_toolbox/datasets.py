# coding: utf-8

from __future__ import annotations

import typing

import numpy as np
import tensorflow as tf

if typing.TYPE_CHECKING:
    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = ['generate_dataset_from_data']


def generate_dataset_from_data(datas: typing.List[ArrayLike],
                               labels: ArrayLike,
                               batch_size: int = 32,
                               prefetch: bool = True) -> tf.data.Dataset:
    """ Concatenates multiple data arrays into a single dataset, and set it up to perform optimally
        when running with both a CPU and a GPU.

        Args:
            datas (List[ArrayLike]): The data arrays to concatenate.
            labels (ArrayLike): The labels.
            batch_size (int): The batch size.
            prefetch (bool): Whether to prefetch the dataset.

        Returns:
            (tf.data.Dataset) The concatenated dataset.
    """
    concatenated_data = tf.data.Dataset.from_tensor_slices(tuple(datas))
    labels_data = tf.data.Dataset.from_tensor_slices(labels)
    concatenated_dataset = tf.data.Dataset.zip((concatenated_data, labels_data)).batch(batch_size)

    if prefetch:
        concatenated_dataset = concatenated_dataset.prefetch(tf.data.AUTOTUNE)

    return concatenated_dataset
