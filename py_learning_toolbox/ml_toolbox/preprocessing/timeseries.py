# coding: utf-8

from __future__ import annotations

import logging
import typing

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if typing.TYPE_CHECKING:
    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = [
    'get_future_dates',
    'get_labeled_windows',
    'make_future_forecasts',
    'make_train_test_split',
    'make_windows',
]


def get_labeled_windows(windows: tf.Tensor, horizon: int = 1) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """ Create labels for windowed dataset.

        E.g. if horizon=1
            Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])

        Args:
            windows (tf.Tensor): the data to generate the window and horizon for.
            horizon (int): the size of the output length list.

        Returns:
            (typing.Tuple) the separated window and horizon of the data.
    """
    return windows[:, :-horizon], windows[:, -horizon:]


def make_windows(data: tf.Tensor, window_size: int = 7, horizon: int = 1) -> typing.Tuple[typing.List, typing.List]:
    """ Turns a 1D array into a 2D array of sequential labeled windows of
        window_size with horizon size labels.

        Args:
            data (tf.Tensor): the data to make windows from.
            window_size (int): the size of the window.
            horizon (int): the size of the horizon.

        Returns:
            (typing.Tuple) 2D array of labeled windowed dataset.
    """
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)

    # create 2D array of windows of size, window_size
    window_indexes = window_step + np.expand_dims(np.arange(len(data) - (window_size + horizon - 1)), axis=0).T

    windowed_array = data[window_indexes]

    # Get labeled windows
    return get_labeled_windows(windowed_array, horizon=horizon)


def make_train_test_split(data: tf.Tensor, labels: tf.Tensor, test_split: float = 0.2) -> typing.Tuple:
    """ Splits matching pairs of data and labels into train and test splits.

        Args:
            data (tf.Tensor): the data to split.
            labels (tf.Tensor): the labels corresponding to the windowed data
            test_split (float): the fraction of data dedicated to be used as test set,
                must fall between 0-1.

        Raises:
            AssertionError: when test_split does not fall between 0-1.

        Returns
            (typing.Tuple) the train-test split of the windows and labels
                -> (X_train, X_test, y_train, y_test).
    """
    assert (test_split >= 0 and test_split <= 1), 'test_split must be between 0-1'

    split_index = int(len(data) * (1 - test_split))

    X_train = data[:split_index]
    y_train = labels[:split_index]
    X_test = data[split_index:]
    y_test = labels[split_index:]

    return X_train, X_test, y_train, y_test


def make_future_forecasts(values: tf.data.Dataset,
                          model: tf.keras.models.Model,
                          into_future: int,
                          window_size: int,
                          log_step: bool = False) -> typing.List[float]:
    """ Make future forecasts for a set number of steps.

        Args:
            values (tf.data.Dataset): the values to make forecasts on.
            model (tf.keras.models.Model): the model to make the predictions with.
            into_future (int): how far into the future to make predictions.
            window_size (int): how big of a window to use when making forecasts.
            log_step (bool): whether or not to log the step in the process.

        Returns:
            (typing.List[float]) the future forecasts.
    """
    future_forecast = []
    last_window = values[-window_size:]

    for _ in range(into_future):
        # Making forcasts on own forecast
        future_pred = model.predict(tf.expand_dims(last_window, axis=0), verbose=0)
        pred = tf.squeeze(future_pred).numpy()

        if log_step:
            logger.info(f'Predicting On: \n {last_window} -> Prediction: {pred}')

        future_forecast.append(pred)

        last_window = np.append(last_window, future_pred)[-window_size:]

    return future_forecast


def get_future_dates(start_date: np.datetime64, into_future: int = 7, offset: int = 1) -> np.ndarray[np.datetime64]:
    """ Generates the following dates from the start date for a number of days into the future.

        Example:
            get_future_dates('7/12/2022', 3, 3) -> ['7/15/2022', '7/16/2022', '7/17/2022']

        Args:
            start_date (np.datetime64): the starting date used to determine the future dates from.
            into_future (int): the number of days into the future to generate dates for.
            offset (int): how far to offset into the future from the start_date to start
                the future dates generation.

        Returns:
            (np.ndarray[np.datetime64]) the future dates offset from the starting date.
    """
    start_date = start_date + np.timedelta64(offset, 'D')
    end_date = start_date + np.timedelta64(into_future, 'D')

    return np.arange(start_date, end_date, dtype='datetime64[D]')
