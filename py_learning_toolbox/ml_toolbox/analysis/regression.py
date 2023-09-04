# coding: utf-8
from __future__ import annotations

from dataclasses import dataclass
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

if typing.TYPE_CHECKING:
    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = [
    'generate_prediction_metrics',
    'generate_prediction_metrics_dataframe',
    'plot_predictions',
    'plot_true_versus_predicted',
    'RegressionPredictionMetrics',
]


@dataclass()
class RegressionPredictionMetrics:
    """ A dataclass for regression prediction metrics.
    
        Attributes:
            mae (float): The mean absolute error.
            mse (float): The mean squared error.
            rmse (float): The root mean squared error.
            mape (float): The mean absolute percentage error.
            mase (float): The mean absolute scaled error.
            huber (float): The huber loss.
            name (Optional[str]): The name of the corresponding experiment.
    """
    mae: float
    mse: float
    rmse: float
    mape: float
    mase: float
    huber: float

    name: typing.Optional[str] = None

    def __iter__(self):
        """ Iterates over the dataclass."""
        for key, value in self.__dict__.items():
            yield key, value


def generate_prediction_metrics(y_true: ArrayLike,
                                y_pred: ArrayLike,
                                name: typing.Optional[str] = None) -> RegressionPredictionMetrics:
    """ Evaluates the model predictions using the following metrics:

        - mean absolute error (MAE)
        - mean squared error (MSE)
        - root mean squared error (RMSE)
        - mean absolute percentage error (MAPE)
        - mean absolute scaled error (MASE)
        - huber loss
    
        Args:
            y_true (ArrayLike): The true values.
            y_pred (ArrayLike): The predicted values.
            name (Optional[str]): The name to assign to the metrics

        Returns:
            RegressionPredictionMetrics: The regression prediction metrics.
    """
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy()
    mase = tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy() / np.mean(np.abs(np.diff(y_true)))
    huber = tf.keras.losses.Huber()(y_true, y_pred).numpy()
    return RegressionPredictionMetrics(
        mae=tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy(),
        mse=mse,
        rmse=np.sqrt(mse),
        mape=tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred).numpy(),
        mase=mase,
        huber=huber,
        name=name)


def generate_prediction_metrics_dataframe(
        all_prediction_metrics: typing.List[RegressionPredictionMetrics]) -> pd.DataFrame:
    """ Creates a dataframe of the regression prediction metrics.

        Args:
            all_prediction_metrics (List[RegressionPredictionMetrics]): The prediction metrics.

        Returns:
            (pd.DataFrame) The regression prediction metrics dataframe.
    """
    all_results = {}
    for index, prediction_metrics in enumerate(all_prediction_metrics):
        prediction_name = prediction_metrics.name or f'model_{index}'
        all_results[prediction_name] = dict(prediction_metrics)
        all_results[prediction_name].pop('name')

    return pd.DataFrame(all_results).transpose()


def plot_true_versus_predicted(y_true: ArrayLike,
                               y_predict: ArrayLike,
                               figsize: typing.Optional[typing.Tuple[int, int]] = (10, 7)) -> None:
    """ Plots the actual true values against the predicted values.
        Note that better predictions have a slope closer to 1.

        NOTE: This is a very simple plot, and is not very useful for more complex models.
        NOTE: This only works for regression problems.

        Args:
            y_true (Array): The true values.
            y_predict (Array): The predicted values.
            figsize (Optional[Tuple[int, int]]): The size of the figure.
    """
    plt.figure(figsize=figsize)

    plt.title('Actual Value vs. Predicted Value')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.scatter(y_true, y_predict)


def plot_predictions(X_train: ArrayLike, 
                     y_train_true: ArrayLike,
                     X_test: ArrayLike,
                     y_test_true: ArrayLike,
                     y_test_pred: ArrayLike) -> None:
    """ Plots training data, test data, and compares predictions against actual values.

        NOTE: This is a very simple plot, and is not very useful for more complex models.
        NOTE: This is limited to 1D data.
        NOTE: This only works for regression problems.

        Args:
            X_train (ArrayLike): The training data.
            y_train_true (ArrayLike): The true training labels.
            X_test (ArrayLike): The test data.
            y_test_true (ArrayLike): The true test labels.
            y_test_pred (ArrayLike): The predicted test labels.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, y_train_true, c='b', label='Training Data')
    plt.scatter(X_test, y_test_true, c='g', label='Test Data')
    plt.scatter(X_test, y_test_pred, c='r', label='Predictions')
    plt.legend()
