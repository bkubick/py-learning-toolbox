# coding: utf-8

from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

if typing.TYPE_CHECKING:
    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = ['plot_history', 'plot_histories', 'plot_learning_rate_versus_loss']


def plot_history(history: typing.Union[tf.keras.callbacks.History, dict, pd.DataFrame],
                 metric: typing.Optional[str] = None) -> None:
    """ Plots the loss versus the epochs.

        Raises:
            TypeError: If the type of history is not supported.

        Args:
            history (tf.keras.callbacks.History): The history object returned by the fit method.
            metric (Optional[str]): The metric to plot.
    """
    if isinstance(history, tf.keras.callbacks.History):
        history_df = pd.DataFrame(history.history)
    elif isinstance(history, dict):
        history_df = pd.DataFrame(history)
    elif isinstance(history, pd.DataFrame):
        history_df = history
    else:
        raise TypeError(f'Invalid type for history: {type(history)}')

    if metric:
        metrics = [metric]
        val_metric = f'val_{metric}'
        if val_metric in history_df.columns.to_list():
            metrics.append(val_metric)
        history_df.loc[:, metrics].plot()
        plt.ylabel(metric.capitalize())
        plt.xlabel('Epochs')
        plt.title(f'{metric.capitalize()} vs. Epochs')
    else:
        history_df.plot()
        plt.ylabel('Metrics')
        plt.xlabel('Epochs')
        plt.title('Metrics vs. Epochs')
        plt.show()


def plot_histories(original_history: typing.Union[pd.DataFrame, tf.keras.callbacks.History],
                   new_history: typing.Union[pd.DataFrame, tf.keras.callbacks.History],
                   initial_epoch: int,
                   metric: typing.Optional[str] = None) -> None:
    """ Compares the history of two models where the second model's starting epoch is the first model's
        ending epoch. This is useful for comparing the history of a model before and after fine-tuning.
        
        This plots 'loss' and 'val_loss' versus the epochs for both models, and the metric and validation metric
        versus the epochs for both models.
    
        Args:
            original_history (Union[pd.DataFrame, tf.keras.callbacks.History]): The history of the original model.
            new_history (Union[pd.DataFrame, tf.keras.callbacks.History]): The history of the new model.
            initial_epoch (int): The epoch at which the new model started training at.
            metric (Optional[str]): The metric to plot (defaults to 'accuracy').
    """
    if isinstance(original_history, pd.DataFrame) and isinstance(new_history, pd.DataFrame):
        original_history_df = original_history
        new_history_df = new_history
    else:
        original_history_df = pd.DataFrame(original_history.history)
        new_history_df = pd.DataFrame(new_history.history)

    metric_to_plot = metric or 'accuracy'

    total_acc = pd.concat([original_history_df[metric_to_plot], new_history_df[metric_to_plot]])
    total_loss = pd.concat([original_history_df['loss'], new_history_df['loss']])
    total_val_acc = pd.concat([original_history_df[f'val_{metric_to_plot}'], new_history_df[f'val_{metric_to_plot}']])
    total_val_loss = pd.concat([original_history_df['val_loss'], new_history_df['val_loss']])
    
    # Loss Plots
    plt.figure(figsize=(8,8))
    plt.subplot(2, 1, 1)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epoch-1, initial_epoch-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Accuracy Plots
    plt.figure(figsize=(8,8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label=f'Training {metric_to_plot.capitalize()}')
    plt.plot(total_val_acc, label=f'Validation {metric_to_plot.capitalize()}')
    plt.plot([initial_epoch-1, initial_epoch-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation {metric_to_plot.capitalize()}')


def plot_learning_rate_versus_loss(learning_rates: typing.List[float],
                                   losses: typing.List[float],
                                   figsize: typing.Tuple[int, int] = (10, 7)) -> None:
    """ Plots the loss versus the learning rate for each epoch.

        Args:
            learning_rates (List[float]): The learning rates.
            losses (List[float]): The losses.
            figsize (Tuple[int, int]): The size of the figure.
    """
    plt.figure(figsize=figsize)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs. Loss')
    plt.plot(learning_rates, losses)
    plt.xscale('log')
