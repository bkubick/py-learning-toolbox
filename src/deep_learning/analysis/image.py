# coding: utf-8

from __future__ import annotations

import logging
import math
import os
import random
import typing

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if typing.TYPE_CHECKING:
    import pathlib

    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = [
    'get_classnames_from_directory',
    'load_and_resize_image',
    'plot_image',
    'plot_images',
    'plot_random_image_label_and_prediction',
    'summarize_directory',
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def summarize_directory(directory: pathlib.Path) -> None:
    """ Summarizes the number of images in each directory of the data directory.

        Args:
            data_directory: The directory containing the data.
    """
    for dirpath, _, filenames in os.walk(directory):
        images = [file for file in filenames if file.endswith('jpg') or file.endswith('jpeg') or file.endswith('png')]
        if images:
            logger.info(f'Directory: {dirpath} Total Images: {len(images)}')


def get_classnames_from_directory(directory: pathlib.Path) -> np.ndarray:
    """ Gets the class names from the data directory.

        Args:
            data_directory (pathlib.Path): The directory containing the data.

        Returns:
            (np.ndarray) The class names.
    """
    all_class_names = [
        item.name for item in directory.iterdir() if item.is_dir() and not item.name.startswith('.')
    ]
    class_names = np.array(sorted(all_class_names))
    return class_names


def plot_image(index: int, images: ArrayLike, labels: ArrayLike, class_names: typing.List[str], black_and_white: bool = False):
    """ Plots an image from the dataset.

        Args:
            index (int): The index of the image to plot.
            images (Array): The data to plot.
            labels (Array): The labels of the data.
            class_names (List[str]): The names of the classes.
            black_and_white (bool): Whether to plot the image in black and white.
    """
    cmap = None
    if black_and_white:
        cmap = plt.cm.binary

    plt.imshow(images[index], cmap=cmap)
    plt.title(class_names[labels[index]])


def plot_images(indexes: typing.List[int],
                images: ArrayLike,
                labels: ArrayLike,
                class_names: typing.List[str],
                black_and_white: bool = False):
    """ Plots each image from the dataset at each corresponding index.

        Raises:
            AssertionError: If the number of indexes is greater than 4.

        Args:
            indexes (List[int]): The index of the image to plot.
            images (Array): The data to plot.
            labels (Array): The labels of the data.
            class_names (List[str]): The names of the classes.
            black_and_white (bool): Whether to plot the image in black and white.
    """
    assert len(indexes) <= 4, 'Cannot plot more than 4 images at a time.'

    cmap = None
    if black_and_white:
        cmap = plt.cm.binary

    total_images_per_row: int = 4
    total_rows: int = math.ceil(len(indexes) / total_images_per_row)

    _, axes = plt.subplots(total_rows, min([total_images_per_row, len(indexes)]), figsize=(15, 7))
    for i, fig_index in enumerate(indexes):
        axes[i].imshow(images[fig_index], cmap=cmap)
        axes[i].set_title(class_names[labels[fig_index]])


def plot_random_image_label_and_prediction(images: ArrayLike,
                                           true_labels: ArrayLike,
                                           pred_probabilities: ArrayLike,
                                           class_names: typing.List[str],
                                           black_and_white: bool = False):
    """ Plots a random image from the dataset.

        NOTE: images and labels must be the same length.
        NOTE: class_names must be the same length as the number of classes in the dataset.

        Args:
            images (Array): The data to plot.
            true_labels (Array): The true labels of the data.
            pred_probabilities (Array): The predicted probabilities of the data.
            class_names (List[str]): The names of the classes.
            black_and_white (bool): Whether to plot the image in black and white.
    """
    index_of_choice = random.randint(0, len(images))

    target_image = images[index_of_choice]

    pred_label = class_names[pred_probabilities[index_of_choice].argmax()]
    true_label = class_names[true_labels[index_of_choice]]

    cmap = None
    if black_and_white:
        cmap = plt.cm.binary

    plt.imshow(target_image, cmap=cmap)

    x_label_color = 'green' if pred_label == true_label else 'red'
    plt.xlabel(f'Pred: {pred_label}  True: {true_label}', color=x_label_color)
