# coding: utf-8
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import tensorflow as tf


__all__ = ['generate_tensorboard_callback', 'generate_checkpoint_callback', 'generate_csv_logger_callback']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_tensorboard_callback(experiment_name: str,
                                  filepath: Optional[str] = None,
                                  include_timestamp: bool = True) -> tf.keras.callbacks.TensorBoard:
    """ Creates a TensorBoard callback.

        Args:
            experiment_name (str): The experiment name.
            filepath (Optional[str]): The directory name.
            include_timestamp (bool): Whether to include the timestamp in the filename.

        Returns:
            (tf.keras.callbacks.TensorBoard) The TensorBoard callback.
    """
    log_dir = f'{filepath or "logs"}/{experiment_name}'
    if include_timestamp:
        log_dir = f'{log_dir}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    logger.info(f'TensorBoard callback for {log_dir}')

    return tensorboard_callback


def generate_checkpoint_callback(experiment_name: str,
                                 filepath: Optional[str] = None,
                                 monitor: Optional[str] = None,
                                 best_only: bool = True,
                                 include_timestamp: bool = True) -> tf.keras.callbacks.ModelCheckpoint:
    """ Generates a checkpoint callback.

        Args:
            experiment_name (str): The experiment name.
            filepath (Optional[str]): The directory name.
            monitor (Optional[str]): The metric to monitor.
            best_only (bool): Whether to save only the best model.
            include_timestamp (bool): Whether to include the timestamp in the filename.
        
        Returns:
            (tf.keras.callbacks.ModelCheckpoint) The checkpoint callback.
    """
    log_dir = f'{filepath or "checkpoints"}/{experiment_name}'
    if include_timestamp:
        log_dir = f'{log_dir}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir,
        monitor=monitor or 'val_accuracy',
        save_weights_only=True,
        save_best_only=best_only,
        save_freq='epoch',
        verbose=1)

    logger.info(f'Checkpoint callback for {log_dir}')

    return checkpoint


def generate_csv_logger_callback(experiment_name: str,
                                 filepath: Optional[str] = None,
                                 filename: Optional[str] = None,
                                 include_timestamp: bool = True) -> tf.keras.callbacks.CSVLogger:
    """ Generates a CSV logger callback.
    
        Args:
            experiment_name (str): The experiment name.
            filepath (Optional[str]): The directory name.
            filename (str): The filename of the CSV file (defaults to "epoch_results.csv").
            include_timestamp (bool): Whether to include the timestamp in the filename.
        
        Returns:
            (tf.keras.callbacks.CSVLogger) The CSV logger callback.
    """
    log_dir = f'{filepath or "logs"}/{experiment_name}'
    if include_timestamp:
        log_dir = f'{log_dir}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    if filename and not filename.endswith('.csv'):
        filename = f'{filename}.csv'

    csv_logger = tf.keras.callbacks.CSVLogger(f'{log_dir}/{filename or "epoch_results.csv"}')

    logger.info(f'CSV Logger callback for {log_dir}')

    return csv_logger
