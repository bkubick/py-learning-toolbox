# coding: utf-8

from __future__ import annotations

import logging
import os
import typing

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


__all__ = ['export_and_verify_model', 'plot_model']


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


def export_and_verify_model(model: tf.keras.models.Model,
                            validation_data: typing.Union[tf.data.Dataset, tf.Tensor, np.ndarray, typing.List],
                            loss: typing.Union[str, typing.Callable],
                            strip_pruning: bool = False,
                            file_type: typing.Optional[str] = None,
                            filepath: typing.Optional[str] = None,
                            custom_objects: typing.Optional[typing.Dict[str, typing.Any]] = None,
                            metrics: typing.Optional[typing.List[str]] = None,
                            optimizer: typing.Optional[typing.Union[str, typing.Callable]] = None,
                            tolerance: float = 1e-08) -> bool:
    """ Exports a model and verifies that it can be loaded and evaluated to 'close' to the original model.

        NOTE: Model will save to '{filepath}/{model name"}'.

        Args:
            model (tf.keras.models.Model): the model to export.
                The model name will be used as the filename if defined, otherwise 'model' will be used.
            validation_data (Union[tf.data.Dataset, tf.Tensor, np.ndarray, List]): the validation data to evaluate
                the model on.
            loss (Union[str, Callable]): the loss function to use when compiling the model.
            strip_pruning (bool): whether or not to strip pruning from the model before saving.
            file_type (Optional[str]): the file type to save the model as. Defaults to None.
            filepath (Optional[str]): the file path to save the model to. Defaults to './models/'.
            custom_objects (Optional[Dict]): the custom layers to pass to tf.keras.models.load_model. Defaults to None.
            metrics (Optional[List[str]]): the metrics to use when compiling the model. Defaults to None.
            optimizer (Optional[Union[str, Callable]]): the optimizer to use when compiling the model.
                Defaults to "Adam".
            tolerance (float): the tolerance to use when comparing the original model's evaluation metrics to the
                loaded model's evaluation metrics. Defaults to 1e-08.
        
        Returns:
            (bool) whether or not the exported model can be loaded and evaluated to 'close' to the original model.
    """
    filepath = f'{filepath or "./models"}/{model.name or "model"}'
    filepath = f'{filepath}.{file_type}' if file_type == 'h5' else filepath

    # Last name in filepath is the filename
    dirs = '/'.join(filepath.split('/')[:-1])
    if not os.path.exists(dirs):
        logging.info(f'Creating directories: {dirs}')
        os.makedirs(dirs)

    model_eval = model.evaluate(validation_data, verbose=0)
    logging.info(f'Original Model Evaluation Metrics: {model_eval}')

    if strip_pruning:
        model_export = tfmot.sparsity.keras.strip_pruning(model)
    else:
        model_export = model

    logging.info(f'Saving Model to: {filepath}')
    model_export.save(filepath, save_format=file_type)

    # Loading and evaluating exported model
    logging.info(f'Loading Model from: {filepath}')
    model_loaded: tf.keras.models.Model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
    model_loaded.compile(loss=loss,
                         optimizer=optimizer or tf.keras.optimizers.legacy.Adam(),
                         metrics=metrics or [])

    model_loaded_eval = model_loaded.evaluate(validation_data, verbose=0)
    logging.info(f'Loaded Model Evaluation Metrics: {model_loaded_eval}')

    return np.allclose(model_eval, model_loaded_eval, equal_nan=True, atol=tolerance)
