# coding: utf-8

from __future__ import annotations

import typing

import tensorflow as tf


__all__ = ['plot_model']


def plot_model(model: tf.keras.Model) -> typing.Any:
    """ Visualizes the model structure (layers and their neurons).

        NOTE: Only works if pydot and graphviz are installed, and on a Jupyter notebook.

        Args:
            model: The model to visualize.

        Returns:
            (tf.keras.utils.vis_utils.ModelVisualization) The model visualization.
    """
    return tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
