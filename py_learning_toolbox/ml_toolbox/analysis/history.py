# coding: utf-8

from __future__ import annotations

import datetime as dt
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

if typing.TYPE_CHECKING:
    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = [
    'export_history',
    'import_history',
    'plot_history',
    'plot_learning_rate_versus_loss',
    'plot_sequential_histories',
]


_CHECKPOINT_COLORS = [
    '#C5DFFF',
    '#98C7FF',
    '#5CA7FF',
    '#2187FF',
    '#0061D2',
    '#004392',
    '#002B5C',
    '#000000',
]


def export_history(history: typing.Union[tf.keras.callbacks.History, dict, pd.DataFrame],
                   experiment_name: str,
                   filepath: typing.Optional[str] = None,
                   file_format: typing.Optional[str] = None,
                   include_timestamp: bool = True) -> None:
    """ Exports the history to a file of either .json or .csv under the experiment directory.

        Note: If the filepath is not specified, the history will be saved to the 'logs' directory.
    
        Raises:
            TypeError: If the type of history is not supported.
            ValueError: If the file format is not supported.
            
        Args:
            history (Union[tf.keras.callbacks.History, dict, pd.DataFrame]): The history object returned
                by the fit method.
            experiment_name (str): The experiment name.
            filepath (Optional[str]): The filepath to save the history to.
            file_format (Optional[str]): The file format to save the history to.
                Must be one of the following: ['json', 'csv'].
            include_timestamp (bool): Whether to include the timestamp in the filepath.
    """
    file_format = file_format or 'csv'
    if file_format not in {'json', 'csv'}:
        raise ValueError(f'Invalid file format: {file_format}')

    log_dir = f'{filepath or "logs"}/{experiment_name}'
    if include_timestamp:
        log_dir = f'{log_dir}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    if isinstance(history, tf.keras.callbacks.History):
        history_df = pd.DataFrame(history.history)
    elif isinstance(history, dict):
        history_df = pd.DataFrame(history)
    elif isinstance(history, pd.DataFrame):
        history_df = history
    else:
        raise TypeError(f'Invalid type for history: {type(history)}')

    filename = f'history.{file_format}'
    if file_format == 'json':
        history_df.to_json(f'{log_dir}/{filename}')
    else:
        history_df.to_csv(f'{log_dir}/{filename}')


def import_history(filepath: str) -> pd.DataFrame:
    """ Imports the history from a file of either .json or .csv and converts to pd dataframe.
    
        Raises:
            ValueError: If the filepath is invalid.
            
        Args:
            filepath (str): The filepath to save the history to.
            
        Returns:
            pd.DataFrame: The history.
    """
    if filepath.endswith('.json'):
        history = pd.read_json(filepath)
    elif filepath.endswith('.csv'):
        history = pd.read_csv(filepath)
    else:
        raise ValueError(f'Invalid filepath: {filepath}')

    return history


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


def plot_sequential_histories(histories: typing.List[typing.Union[pd.DataFrame, tf.keras.callbacks.History]],
                              metric: typing.Optional[str] = None,
                              figsize: typing.Tuple[int, int] = (8, 8)) -> None:
    """ Plots histories of a model with multiple training sessions where the starting epoch is the previous history's
        ending epoch. This is useful for comparing the history of a model before and after fine-tuning.
        
        This plots 'loss' and 'val_loss' versus the epochs for the model histories, and the metric and
        validation metric versus the epochs for all histories.
    
        Args:
            histories (List[Union[pd.DataFrame, tf.keras.callbacks.History]]): The histories from the
                training sessions performed on the same model.
                NOTE: This is typically useful when halting training and fine-tuning a model with additional
                epochs after adjusting the learning rate or freezing certain parameters.
            metric (Optional[str]): The metric to plot (defaults to 'loss').
            figsize (Tuple[int, int]): The size of the figure.
    """
    metric_to_plot = metric or 'loss'

    all_history_dfs = []
    for history in histories:
        if isinstance(history, pd.DataFrame):
            history_df = history
        else:
            history_df = pd.DataFrame(history.history)
        all_history_dfs.append(history_df)
    all_history_df = pd.concat(all_history_dfs)

    total_metric = all_history_df[metric_to_plot].to_numpy()
    total_val_metric = None
    if f'val_{metric_to_plot}' in all_history_df:
        total_val_metric = all_history_df[f'val_{metric_to_plot}'].to_numpy()

    plt.figure(figsize=figsize)
    plt.title(f'{metric_to_plot.capitalize()} Checkpoint History')
    plt.xlabel('Epochs')
    plt.ylabel(metric_to_plot.capitalize())
    
    plt.plot(total_metric, label=f'Training {metric_to_plot.capitalize()}')
    if total_val_metric is not None:
        plt.plot(total_val_metric, label=f'Validation {metric_to_plot.capitalize()}')

    checkpoint_epoch = 0
    for index, history in enumerate(all_history_dfs):
        if index > 0:
            plt.axvline(checkpoint_epoch - 1, c=_CHECKPOINT_COLORS[index - 1 % len(_CHECKPOINT_COLORS)], label=f'Checkpoint {index}')

        checkpoint_epoch += len(history)

    plt.legend(loc='lower right')


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
