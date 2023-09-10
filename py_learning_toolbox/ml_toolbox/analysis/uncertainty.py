# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass

import scipy.stats as stats
import tensorflow as tf


__all__ = ['get_uncertainty', 'UncertaintyMetrics']


@dataclass()
class UncertaintyMetrics:
    """ A dataclass for uncertainty metrics.
    
        Attributes:
            mean (tf.Tensor): The mean of the predictions.
            std_dev (tf.Tensor): The standard deviation of the predictions.
            confidence_percentile (float): The confidence percentile.
            lower (tf.Tensor): The lower bound of the confidence interval.
            upper (tf.Tensor): The upper bound of the confidence interval.
    """
    mean: tf.Tensor
    std_dev: tf.Tensor
    confidence_percentile: float
    lower: tf.Tensor
    upper: tf.Tensor


def get_uncertainty(predictions: tf.Tensor, percentile: float = 0.95) -> UncertaintyMetrics:
    """ Generates the prediction upper and lower bounds based of the designated confidence percentile.

        Args:
            predictions (tf.Tensor): the predictions to generate confidence range on.
            percentile (float): the percentile level of confidence to create a range from.

        Returns:
            (UncertaintyMetrics) the lower and upper bounds for each prediction.
    """
    std_dev = tf.math.reduce_std(predictions, axis=0)
    preds_mean = tf.reduce_mean(predictions, axis=0)
    interval = stats.norm.ppf(1 - (1 - percentile) / 2) * std_dev

    lower = preds_mean - interval
    upper = preds_mean + interval

    return UncertaintyMetrics(
        mean=tf.constant(preds_mean),
        std_dev=tf.constant(std_dev),
        confidence_percentile=percentile,
        lower=tf.constant(lower),
        upper=tf.constant(upper))
