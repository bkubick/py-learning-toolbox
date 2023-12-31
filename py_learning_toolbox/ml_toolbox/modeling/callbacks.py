# coding: utf-8

from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Optional

import tensorflow as tf


__all__ = ['generate_tensorboard_callback', 'generate_checkpoint_callback', 'generate_csv_logger_callback']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_tensorboard_callback(model_name: str,
                                  filepath: Optional[str] = None,
                                  include_timestamp: bool = True) -> tf.keras.callbacks.TensorBoard:
    """ Creates a TensorBoard callback.

        - Stores file at {filepath}/{model_name}/tensorboard/{timestamp}

        Args:
            model_name (str): The experiment name.
            filepath (Optional[str]): The directory name.
                Defaults to 'logs'
            include_timestamp (bool): Whether to include the timestamp in the filename.

        Returns:
            (tf.keras.callbacks.TensorBoard) The TensorBoard callback.
    """
    log_dir = f'{filepath or "logs"}/{model_name}/tensorboard'
    if include_timestamp:
        log_dir = f'{log_dir}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    logger.info(f'TensorBoard callback for {log_dir}')

    return tensorboard_callback


def generate_checkpoint_callback(model_name: str,
                                 filepath: Optional[str] = None,
                                 monitor: Optional[str] = None,
                                 best_only: bool = True,
                                 include_timestamp: bool = True) -> tf.keras.callbacks.ModelCheckpoint:
    """ Generates a checkpoint callback.

        - Stores file at {filepath}/{model_name}/{timestamp}/checkpoint.ckpt

        Args:
            model_name (str): The experiment name.
            filepath (Optional[str]): The directory name.
                Defaults to 'checkpoints'
            monitor (Optional[str]): The metric to monitor.
            best_only (bool): Whether to save only the best model.
            include_timestamp (bool): Whether to include the timestamp in the filename.
        
        Returns:
            (tf.keras.callbacks.ModelCheckpoint) The checkpoint callback.
    """
    log_dir = f'{filepath or "checkpoints"}/{model_name}'
    if include_timestamp:
        log_dir = f'{log_dir}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{log_dir}/checkpoint.ckpt',
        monitor=monitor or 'val_accuracy',
        save_weights_only=True,
        save_best_only=best_only,
        save_freq='epoch',
        verbose=1)

    logger.info(f'Checkpoint callback for {log_dir}')

    return checkpoint


def generate_csv_logger_callback(model_name: str,
                                 filepath: Optional[str] = None,
                                 include_timestamp: bool = True) -> tf.keras.callbacks.CSVLogger:
    """ Generates a CSV logger callback.

        - Stores file at {filepath}/{model_name}/csv/{timestamp}/epoch_results.csv

        Args:
            model_name (str): The model name.
            filepath (Optional[str]): The parent directory for the model_name dir with the csv files.
                Defaults to 'logs'
            include_timestamp (bool): Whether to include the timestamp in the filename.

        Returns:
            (tf.keras.callbacks.CSVLogger) The CSV logger callback.
    """
    log_dir = f'{filepath or "logs"}/{model_name}/csv'
    if include_timestamp:
        log_dir = f'{log_dir}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    csv_logger = tf.keras.callbacks.CSVLogger(f'{log_dir}/epoch_results.csv')

    logger.info(f'CSV Logger callback for {log_dir}')

    return csv_logger
