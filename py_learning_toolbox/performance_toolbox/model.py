# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
import time
import typing

if typing.TYPE_CHECKING:
    import numpy as np
    from sklearn.pipeline import Pipeline
    import tensorflow as tf

    Dataset = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray, tf.data.Dataset]
    ImplementsPredict = typing.Union[tf.keras.Model, Pipeline]


__all__ = ['ModelPerformance', 'prediction_timer']


@dataclass()
class ModelPerformance:
    """ A dataclass for storing performance metrics.
    
        Args:
            total_time (float): The total time.
            total_predictions (int): The total number of predictions.
    """
    total_time: float
    total_predictions: int

    @property
    def time_per_prediction(self) -> float:
        """ The time per prediction."""
        return self.total_time / self.total_predictions


def prediction_timer(model: ImplementsPredict, data_to_predict: Dataset) -> ModelPerformance:
    """ Timer for how long a model takes to make predictions on samples.

        Args:
            model (tf.keras.Model): The model to run performance metrics on.
            data_to_predict (Dataset): The data to make predictions on.

        Returns:
            (ModelPerformance) The model performance.        
    """
    start_time = time.perf_counter()
    _ = model.predict(data_to_predict)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    return ModelPerformance(total_time=total_time, total_predictions=len(data_to_predict))
