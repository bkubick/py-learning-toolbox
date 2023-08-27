# coding: utf-8
import tensorflow as tf


__all__ = ['plot_model']


def plot_model(model: tf.keras.Model) -> tf.keras.utils.vis_utils.ModelVisualization:
    """ Visualizes the model structure (layers and their neurons).

        NOTE: Only works if pydot and graphviz are installed, and on a Jupyter notebook.

        Args:
            model: The model to visualize.
    """
    return tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
