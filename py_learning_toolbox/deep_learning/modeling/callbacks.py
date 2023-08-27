# coding: utf-8
import datetime as dt
import logging
from typing import Optional

import tensorflow as tf


__all__ = ['generate_tensorboard_callback', 'generate_checkpoint_callback', 'generate_csv_logger_callback']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_tensorboard_callback(experiment_name: str,
                                  log_path: Optional[str] = None) -> tf.keras.callbacks.TensorBoard:
    """ Creates a TensorBoard callback.

        Args:
            experiment_name (str): The experiment name.
            log_path (Optional[str]): The directory name.

        Returns:
            (tf.keras.callbacks.TensorBoard) The TensorBoard callback.
    """
    log_dir = f"{log_path or 'logs'}/{experiment_name}/{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    logger.info('TensorBoard callback created.')
    
    return tensorboard_callback


def generate_checkpoint_callback(checkpoint_path: str,
                                 monitor: Optional[str] = None,
                                 best_only: bool = True) -> tf.keras.callbacks.ModelCheckpoint:
    """ Generates a checkpoint callback.

        Args:
            checkpoint_path (str): The path to save the checkpoint to.
            monitor (Optional[str]): The metric to monitor.
            best_only (bool): Whether to save only the best model.
        
        Returns:
            (tf.keras.callbacks.ModelCheckpoint) The checkpoint callback.
    """
    if monitor is None:
        monitor = 'val_accuracy'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path or 'checkpoints',
        monitor=monitor,
        save_weights_only=True,
        save_best_only=best_only,
        save_freq='epoch',
        verbose=1)

    return checkpoint


def generate_csv_logger_callback(filename: str, logs_dir: Optional[str] = None) -> tf.keras.callbacks.CSVLogger:
    """ Generates a CSV logger callback.
    
        Args:
            filename (str): The filename of the CSV logger.
            logs_dir (Optional[str]): The directory to save the CSV logger to.
        
        Returns:
            (tf.keras.callbacks.CSVLogger) The CSV logger callback.
    """
    if logs_dir is None:
        logs_dir = 'logs'

    csv_logger = tf.keras.callbacks.CSVLogger(f'{logs_dir}/{filename}')
    return csv_logger
