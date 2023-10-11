# coding: utf-8

from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


__all__ = ['make_future_forecasts', 'plot_price_vs_timesteps']


def plot_price_vs_timesteps(timesteps: np.ndarray[np.datetime64],
                            values: typing.Union[np.ndarray, tf.Tensor, list],
                            format: str = '.',
                            start: int = 0,
                            end: typing.Optional[int] = None,
                            label: typing.Optional[str] = None,
                            title: typing.Optional[str] = None,
                            new_figure: bool = False) -> None:
    """ Plots the timeseries data.

        Args:
            timesteps (np.ndarray[np.datetime64]): the timesteps associated with the timeseries
                data to be plotted on the x-axis.
            values (typing.Union[np.ndarray, tf.Tensor, list]): the pricing data for the timeseries
                to be plotted on the y-axis.
            format (str): the type of format to use for the plot ('.', '-', 'g', etc...)
            start (int): the starting point of the plot (used to "zoom in" the plot)
            end (int): the end point of the plot (used to "zoom in" the plot)
            label (str): label of the plot (used in legend).
            title (str): title the plot.
            new_figure (bool): whether or not to generate a new figure when called.
    """
    if new_figure:
        plt.figure(figsize=(10, 7))

    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel('Time')
    plt.ylabel('Price ($)')

    if title:
        plt.title(title)

    if label:
        plt.legend(fontsize=14)

    plt.grid(True)


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
