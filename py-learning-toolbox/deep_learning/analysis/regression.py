# coding: utf-8
from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if typing.TYPE_CHECKING:
    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = ['plot_true_versus_predicted']


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
