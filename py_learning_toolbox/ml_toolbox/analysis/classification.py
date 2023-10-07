# coding: utf-8

from __future__ import annotations

import itertools
import typing

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
import tensorflow as tf

if typing.TYPE_CHECKING:
    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = [
    'ClassificationPredictionMetrics',
    'generate_prediction_metrics',
    'generate_prediction_metrics_dataframe',
    'generate_prediction_metrics_from_dataset_and_model',
    'generate_prediction_metrics_from_dataset_and_models',
    'plot_confusion_matrix',
    'plot_classification_report',
]


@dataclass()
class ClassificationPredictionMetrics:
    """ A dataclass for prediction metrics.

        Attributes:
            accuracy (float): The accuracy.
            precision (float): The precision.
            recall (float): The recall.
            f1 (float): The f1 score.
            name (Optional[str]): The name of the corresponding experiment.
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    name: typing.Optional[str] = None

    def __iter__(self):
        """ Iterates over the dataclass."""
        for key, value in self.__dict__.items():
            yield key, value


def generate_prediction_metrics_from_dataset_and_model(
        dataset: tf.data.Dataset,
        model: tf.keras.models.Model) -> ClassificationPredictionMetrics:
    """ Evaluates the model predictions using the following metrics:

        - Accuracy
        - Precision
        - Recall
        - F1

        WARNING: This stores the y_true and y_pred in memory. If the dataset is too large,
        this will cause an OOM error.

        WARNING: This is a slower method of generating the prediction metrics due to it
        having to iterate over the entire dataset in batches and predict on batches one
        at a time.

        NOTE: The purpose of this function is to account for shuffling of the dataset when
        the dataset is batched.

        Args:
            dataset (tf.data.Dataset): The dataset containing the true and predicted labels.
            model (tf.keras.models.Model): The model to evaluate.

        Returns:
            ClassificationPredictionMetrics: The prediction metrics.
    """
    y_true = []
    y_pred = []
    for data, labels in dataset:
        y_true.append(labels)
        y_pred_probs = model.predict(data, verbose=0)

        if y_pred_probs.shape[-1] > 1:
            y_pred.append(tf.argmax(y_pred_probs, axis=1))
        else:
            y_pred.append(tf.cast(tf.round(y_pred_probs), tf.int32))

    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)

    return generate_prediction_metrics(y_true, y_pred, model.name)


def generate_prediction_metrics_from_dataset_and_models(
        dataset: tf.data.Dataset,
        models: typing.List[tf.keras.models.Model],
        name: typing.Optional[str] = None) -> typing.Tuple[ClassificationPredictionMetrics,
                                                           ClassificationPredictionMetrics]:
    """ Evaluates the ensemble of models predictions using the following metrics:

        - Accuracy
        - Precision
        - Recall
        - F1

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
            (ClassificationPredictionMetrics, ClassificationPredictionMetrics): The mean and median prediction metrics, respectively.
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

            if pred_probs.shape[-1] > 1:
                ensemble_preds.append(tf.argmax(pred_probs, axis=1))
            else:
                ensemble_preds.append(tf.squeeze(tf.round(pred_probs)))

        y_mean_preds.append(tf.reduce_mean(ensemble_preds, axis=0))
        y_median_preds.append(np.median(ensemble_preds, axis=0))

    y_true = tf.cast(tf.concat(y_true, axis=0), dtype=tf.int32)
    y_mean_preds = tf.cast(tf.concat(y_mean_preds, axis=0), dtype=tf.int32)
    y_median_preds = tf.cast(tf.concat(y_median_preds, axis=0), dtype=tf.int32)

    return (generate_prediction_metrics(y_true, y_mean_preds, f'{name}_mean'),
            generate_prediction_metrics(y_true, y_median_preds, f'{name}_median'))


def generate_prediction_metrics(y_true: ArrayLike,
                                y_pred: ArrayLike,
                                name: typing.Optional[str] = None) -> ClassificationPredictionMetrics:
    """ Evaluates the model predictions using the following metrics:

        - Accuracy
        - Precision
        - Recall
        - F1

        Args:
            y_true (ArrayLike): The true labels.
            y_pred (ArrayLike): The predicted labels.
            name (Optional[str]): The name to assign to the metrics

        Returns:
            ClassificationPredictionMetrics: The prediction metrics.
    """
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_pred = tf.cast(y_pred, dtype=tf.int32)

    return ClassificationPredictionMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average='weighted'),
        recall=recall_score(y_true, y_pred, average='weighted'),
        f1=f1_score(y_true, y_pred, average='weighted'),
        name=name)


def generate_prediction_metrics_dataframe(
        all_prediction_metrics: typing.List[ClassificationPredictionMetrics]) -> pd.DataFrame:
    """ Creates a dataframe of the prediction metrics.

        Args:
            all_prediction_metrics (List[ClassificationPredictionMetrics]): The prediction metrics.

        Returns:
            (pd.DataFrame) The prediction metrics dataframe.
    """
    all_results = {}
    for index, prediction_metrics in enumerate(all_prediction_metrics):
        prediction_name = prediction_metrics.name or f'model_{index}'
        all_results[prediction_name] = dict(prediction_metrics)
        all_results[prediction_name].pop('name')

    return pd.DataFrame(all_results).transpose()


def plot_confusion_matrix(y_true: ArrayLike,
                          y_pred: ArrayLike,
                          label_text_size: int = 20,
                          cell_text_size: int = 10,
                          classes: typing.Optional[ArrayLike] = None,
                          figsize: typing.Optional[typing.Tuple[int, int]] = (15, 15),
                          norm: bool = False) -> None:
    """ Plots a confusion matrix using Seaborn's heatmap. 

        Args:
            y_true (Array): The true values.
            y_pred (Array): The predicted values.
            label_text_size (int): The size of the labels.
            cell_text_size (int): The size of the text in each cell.
            classes (Optional[List[str]]): The class labels of each category.
            figsize (Optional[Tuple[int, int]]): The size of the figure.
            norm (bool): Whether to display the normalized cells.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, tf.round(y_pred))

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize our confusion matrix
    n_classes = cm.shape[0]

    # Prettifying it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Create classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title='Confusion Matrix',
          xlabel='Predicted Label',
          ylabel='True Label',
          xticks=np.arange(n_classes),
          yticks=np.arange(n_classes),
          xticklabels=labels,
          yticklabels=labels)

    # Make Labels bigger
    ax.yaxis.label.set_size(label_text_size)
    ax.xaxis.label.set_size(label_text_size)
    ax.title.set_size(label_text_size)

    # Make x labels appear on bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Make x labels mostly vertical
    plt.xticks(rotation=70, ha='center')

    # Set the threshold
    threshold = (cm.max() + cm.min()) / 2

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)',
                    horizontalalignment='center',
                    color='white' if cm[i, j] > threshold else 'black',
                    size=cell_text_size)
        else:
            plt.text(j, i, f'{cm[i, j]}',
                    horizontalalignment='center',
                    color='white' if cm[i, j] > threshold else 'black',
                    size=cell_text_size)


def plot_classification_report(y_labels: ArrayLike, y_pred: ArrayLike, class_names: ArrayLike) -> None:
    """ Plots the classification report.

        Args:
            y_labels (ArrayLike[int][int]): The true labels.
            y_pred (ArrayLike[int][int]): The predicted labels.
            class_names (ArrayLike[str]): The class names.
    """
    model_classification_report = classification_report(y_labels, y_pred, output_dict=True)

    # Get the f1 score metric and the corresponding class name
    class_name_to_f1_score = {}
    for class_number, metrics in model_classification_report.items():
        # Multiple non-numeric keys occur which we don't want to store
        try:
            class_number = int(class_number)
        except:
            continue

        class_name=class_names[class_number]
        class_name_to_f1_score[class_name] = metrics['f1-score']

    # Turn to a dataframe
    class_name_to_f1_score_df = pd.DataFrame({
        'class_name': class_name_to_f1_score.keys(),
        'f1_score': class_name_to_f1_score.values()
    })

    # Sort dataframe
    class_name_to_f1_score_df = class_name_to_f1_score_df.sort_values('f1_score', ascending=True)

    # Plotting data
    _, ax = plt.subplots(figsize=(12,25))
    _ = ax.barh(range(len(class_name_to_f1_score_df)), class_name_to_f1_score_df['f1_score'].values)
    ax.set_yticks(range(len(class_name_to_f1_score_df)))
    ax.set_yticklabels(class_name_to_f1_score_df['class_name'])
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score of Predictions for each Class')
