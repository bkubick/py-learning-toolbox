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


def split_data_labels_from_dataset(dataset: tf.data.Dataset,
                                   labels_only: bool = False) -> typing.Union[typing.Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """ Splits the data and labels from the dataset.

        NOTE: This defeats the purpose of using a dataset, but is useful for debugging or extracting
        the data and labels for other purposes.

        WARNING: This will load the entire dataset into memory. If the dataset is too large, this
        will cause an OOM error.

        NOTE: Be caucious of using this function when the dataset is shuffled. The data and labels
        will not be in the same order as the original dataset. Additionally, depending on how the
        dataset is loaded, the data and labels may shuffle everytime the dataset is called, resulting
        in a different order each time.

        Args:
            dataset (tf.data.Dataset): The dataset to split.
            labels_only (bool): Whether to return only the labels. Defaults to False.
                This is useful for when you only need the labels for something like the
                embedding projector, or when the data is too large to load into memory
                when you only need the labels.

        Returns:
            (Union[Tuple[Tensor, Tensor], Tensor]) The data and labels.
    """
    data, labels = [], []
    for X, y in dataset:
        if labels_only is False:
            data.append(X)
        labels.extend(y)

    labels_tensor = tf.convert_to_tensor(labels)
    return (tf.convert_to_tensor(data), labels_tensor) if labels_only is False else labels_tensor
