# coding: utf-8

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import tensorflow as tf

from ... import _utils


if typing.TYPE_CHECKING:
    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = [
    'cyclically_encode_timestamp',
    'get_future_dates',
    'get_labeled_windows',
    'make_windowed_dataset',
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


@_utils.dev.obsolete('make_windows is obsolete. Use make_windowed_dataset instead.')
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


def make_windowed_dataset(series: ArrayLike,
                          window_size: int = 7,
                          horizon_size: int = 1,
                          batch_size: int = 32,
                          buffer_size: typing.Optional[int] = 1000) -> tf.data.Dataset:
    """ Creates a windowed dataset from the given series.

        Example:
            Input: [0, 1, 2, 3, 4, 5, 6, 7]
            Output: [0, 1, 2, 3, 4, 5, 6], [7],
                    [1, 2, 3, 4, 5, 6, 7], [8],
                    ...

        Args:
            series (ArrayLike): the series to create the windowed dataset from.
            window_size (int): the size of the window.
            horizon_size (int): the size of the horizon.
            batch_size (int): the size of the batch.
            buffer_size (Optional[int]): the size of the buffer to shuffle the dataset with.
                When set to None, no shuffling will occur. Defaults to 1000.
        
        Returns:
            (tf.data.Dataset) the windowed dataset.
    """
    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + horizon_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + horizon_size))

    if buffer_size is not None:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.map(lambda window: (window[:-horizon_size], window[-horizon_size:]))
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


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


def cyclically_encode_timestamp(timestamps: typing.Union[typing.List, np.ndarray],
                                time_period: typing.Optional[str] = None) -> tf.Tensor:
    """ Creates a cyclically encoded timestamp based off the period.

        Args:
            timestamps (typing.Union[typing.List, np.ndarray]): the hours of the day to cyclically encode.
            time_period (Optional[str]): the period to encode the hours with. Defaults to 'day_by_week'. Options:
                [`day_by_year`, `day_by_month`, `day_by_week`, `month_by_year`, `hour_by_day`,
                `minute_by_hour`, `second_by_minute`]
                    

        Raises:
            ValueError: if the period is not one of the following:
                - day_by_year
                - day_by_month
                - day_by_week
                - month_by_year
                - hour_by_day
                - minute_by_hour
                - second_by_minute

        Returns:
            (tf.Tensor) the cyclically encoded values for the given period and timestamps.
    """
    period = time_period or 'day_by_week'
    timestamps: pd.DatetimeIndex = pd.to_datetime(timestamps)

    if period == 'day_by_year':
        mapping_func: typing.Callable[[pd.Timestamp], int] = lambda x: x.to_pydatetime().timetuple().tm_yday
        period_steps = timestamps.map(mapping_func)
        max_period_value = 366
    elif period == 'day_by_month':
        period_steps = timestamps.day
        max_period_value = 31
    elif period == 'day_by_week':
        mapping_func: typing.Callable[[pd.Timestamp], int] = lambda x: x.to_pydatetime().timetuple().tm_wday
        period_steps = timestamps.map(mapping_func)
        max_period_value = 7
    elif period == 'month_by_year':
        period_steps = timestamps.month
        max_period_value = 12
    elif period == 'hour_by_day':
        period_steps = timestamps.hour
        max_period_value = 24
    elif period == 'minute_by_hour':
        period_steps = timestamps.minute
        max_period_value = 60
    elif period == 'second_by_minute':
        period_steps = timestamps.second
        max_period_value = 60
    else:
        period_types = ['day_by_year', 'day_by_month', 'day_by_week', 'month_by_year', 'hour_by_day',
                        'minute_by_hour', 'second_by_minute']
        raise ValueError(f'Invalid time_period: {period}: must be one of the following: {period_types}')

    period_tensor = tf.convert_to_tensor(period_steps.to_numpy(), dtype=tf.float32)
    period_sin = tf.math.sin(2 * np.pi * period_tensor / max_period_value)
    period_cos = tf.math.cos(2 * np.pi * period_tensor / max_period_value)

    return tf.stack([period_sin, period_cos], axis=1)
