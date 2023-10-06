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
    'generate_prediction_metrics_from_dataset_and_model',
    'generate_prediction_metrics_from_dataset_and_models',
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


def generate_prediction_metrics_from_dataset_and_model(
        dataset: tf.data.Dataset,
        model: tf.keras.models.Model) -> RegressionPredictionMetrics:
    """ Evaluates the model predictions using the metrics listed below.

        NOTE: This function aggregates the metrics for multi-dimensional values using the mean.

        WARNING: This stores the y_true and y_pred in memory. If the dataset is too large,
        this will cause an OOM error.

        WARNING: This is a slower method of generating the prediction metrics due to it
        having to iterate over the entire dataset in batches and predict on batches one
        at a time.

        NOTE: The purpose of this function is to account for shuffling of the dataset when
        the dataset is batched.

        - mean absolute error (MAE)
        - mean squared error (MSE)
        - root mean squared error (RMSE)
        - mean absolute percentage error (MAPE)
        - mean absolute scaled error (MASE)
        - huber loss

        Args:
            dataset (tf.data.Dataset): The dataset.
            model (tf.keras.models.Model): The model.

        Returns:
            RegressionPredictionMetrics: The regression prediction metrics.
    """
    y_true = []
    y_pred = []

    for X, y in dataset:
        y_true.append(y)
        y_pred.append(model.predict(X, verbose=0))

    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)

    return generate_prediction_metrics(y_true, y_pred, model.name)


def generate_prediction_metrics_from_dataset_and_models(
        dataset: tf.data.Dataset,
        models: typing.List[tf.keras.models.Model],
        name: typing.Optional[str] = None) -> typing.Tuple[RegressionPredictionMetrics,
                                                           RegressionPredictionMetrics]:
    """ Evaluates the ensemble of models predictions using the following metrics:

        - mean absolute error (MAE)
        - mean squared error (MSE)
        - root mean squared error (RMSE)
        - mean absolute percentage error (MAPE)
        - mean absolute scaled error (MASE)
        - huber loss

        NOTE: This function aggregates the metrics for multi-dimensional values using the mean
        for the ensemble mean predictions and the median for the ensemble median predictions.

        WARNING: This stores the y_true and y_pred in memory. If the dataset is too large,
        this will cause an OOM error.

        WARNING: This is a slower method of generating the prediction metrics due to it
        having to iterate over the entire dataset in batches and predict on batches one
        at a time.

        NOTE: The purpose of this function is to account for shuffling of the dataset when
        the dataset is batched.

        Args:
            dataset (tf.data.Dataset): The dataset containing the true and predicted labels.
            models (List[tf.keras.models.Model]): The ensemble models to evaluate.
            name (Optional[str]): The name to assign to the metrics.

        Returns:
            (RegressionPredictionMetrics, RegressionPredictionMetrics): The mean and median
            prediction metrics, respectively.
    """
    name = name or 'ensemble'

    y_true = []
    y_mean_preds = []
    y_median_preds = []
    for data, labels in dataset:
        y_true.append(labels)

        ensemble_preds = []
        for model in models:
            pred_probs = model.predict(data, verbose=0)
            ensemble_preds.append(tf.squeeze(pred_probs))

        y_mean_preds.append(tf.reduce_mean(ensemble_preds, axis=0))
        y_median_preds.append(np.median(ensemble_preds, axis=0))

    y_true = tf.concat(y_true, axis=0)
    y_mean_preds = tf.concat(y_mean_preds, axis=0)
    y_median_preds = tf.concat(y_median_preds, axis=0)

    return (generate_prediction_metrics(y_true, y_mean_preds, f'{name}_mean'),
            generate_prediction_metrics(y_true, y_median_preds, f'{name}_median'))


def generate_prediction_metrics(y_true: ArrayLike,
                                y_pred: ArrayLike,
                                name: typing.Optional[str] = None) -> RegressionPredictionMetrics:
    """ Evaluates the model predictions using the metrics listed below.

        NOTE: This function aggregates the metrics for multi-dimensional values using the mean.

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
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = tf.keras.metrics.mean_absolute_error(y_true, y_pred) / tf.reduce_mean(tf.abs(np.diff(y_true)))
    huber = tf.keras.losses.Huber()(y_true, y_pred)

    # Taking into account multi-dimensional metrics
    if mae.ndim > 0:
        mae = tf.reduce_mean(mae)
    if mse.ndim > 0:
        mse = tf.reduce_mean(mse)
    if rmse.ndim > 0:
        rmse = tf.reduce_mean(rmse)
    if mape.ndim > 0:
        mape = tf.reduce_mean(mape)
    if mase.ndim > 0:
        mase = tf.reduce_mean(mase)
    if huber.ndim > 0:
        huber = tf.reduce_mean(huber)

    return RegressionPredictionMetrics(
        mae=mae.numpy(),
        mse=mse.numpy(),
        rmse=rmse.numpy(),
        mape=mape.numpy(),
        mase=mase.numpy(),
        huber=huber.numpy(),
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
