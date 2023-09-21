# coding: utf-8

from __future__ import annotations

import logging
import typing

import tensorflow as tf


__all__ = ['plot_model']


def plot_model(model: tf.keras.Model, log_info: bool = False) -> typing.Any:
    """ Visualizes the model structure (layers and their neurons). This is a wrapper around
        tf.keras.utils.plot_model, which saves the model architecture to the models directory.

        NOTE: Only works if pydot and graphviz are installed, and on a Jupyter notebook.

        Args:
            model (tf.keras.Model): The model to visualize.

        Returns:
            (tf.keras.utils.vis_utils.ModelVisualization) The model visualization.
    """
    if model.name is None:
        logging.warning('Model does not have a name. Using "model" as the name.')

    name = model.name or 'model'
    filepath = f'./models/{name}_architecture.png'

    if log_info:
        logging.info(f'Saving model architecture to {filepath}')

    return tf.keras.utils.plot_model(model,
                                     show_shapes=True,
                                     show_layer_names=True,
                                     to_file=filepath)
