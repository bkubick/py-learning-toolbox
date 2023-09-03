# coding: utf-8

from __future__ import annotations

import logging
import math
import os
import random
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

if typing.TYPE_CHECKING:
    import pathlib

    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = [
    'get_classnames_from_directory',
    'get_filepaths_from_dataset',
    'get_top_n_mislabeled_images',
    'load_and_resize_image',
    'plot_image_by_index',
    'plot_images_by_indices',
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


def get_filepaths_from_dataset(dataset: tf.data.Dataset, data_filepath: str) -> typing.List[str]:
    """ Gets the image filepaths from the dataset.

        Args:
            dataset (tf.data.Dataset): The dataset.
            data_filepath (str): The filepath to the data.

        Returns:
            (List[str]) The image filepaths.
    """
    image_file_types = ['jpg', 'jpeg', 'png']
    image_file_paths = [f'{data_filepath}/*/*.{file_type}' for file_type in image_file_types]

    filepaths = []
    for filepath in dataset.list_files(image_file_paths, shuffle=False):
        filepaths.append(filepath.numpy())
    
    return filepaths


def get_top_n_mislabeled_images(y_true: ArrayLike,
                                y_pred_probs: ArrayLike,
                                filepaths: typing.List[str],
                                class_names: typing.List[str],
                                n: int = 100) -> pd.DataFrame:
    """ Gets the top n most mislabeled images.

        The DataFrame will contain the following columns:
            - filepath
            - y_true
            - y_pred
            - y_pred_prob
            - y_true_classname
            - y_pred_classname

        Args:
            y_true (ArrayLike): The true labels.
            y_pred_probs (ArrayLike): The predicted probabilities.
            filepaths (List[str]): The filepaths to the images.
            class_names (List[str]): The names of the classes.
            n (int): The number of images to return.

        Returns:
            (pd.DataFrame) The top n most mislabeled images.
    """
    y_pred = y_pred_probs.argmax(axis=1)

    df = pd.DataFrame({
        'filepath': filepaths,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_probs.max(axis=1),
        'y_true_classname': [class_names[i] for i in y_true],
        'y_pred_classname': [class_names[i] for i in y_pred],
    })

    top_n_mislabled_images_df = df[df['y_true'] != df['y_pred']].sort_values('y_pred_prob', ascending=False)[:n]
    return top_n_mislabled_images_df


def plot_image_by_index(index: int,
                        images: ArrayLike,
                        labels: ArrayLike,
                        class_names: typing.List[str],
                        black_and_white: bool = False) -> None:
    """ Plots an image from the dataset by the given index and list of images.

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


def plot_images_by_indices(indexes: typing.List[int],
                           images: ArrayLike,
                           labels: ArrayLike,
                           class_names: typing.List[str],
                           black_and_white: bool = False) -> None:
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
                                           black_and_white: bool = False) -> None:
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