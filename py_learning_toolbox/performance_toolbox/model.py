# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
import time
import typing

import pandas as pd

if typing.TYPE_CHECKING:
    import numpy as np
    from sklearn.pipeline import Pipeline
    import tensorflow as tf

    Dataset = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray, tf.data.Dataset]
    ImplementsPredict = typing.Union[tf.keras.Model, Pipeline]


__all__ = ['generate_performance_dataframe', 'Performance', 'time_predictions']


@dataclass()
class Performance:
    """ A dataclass for storing performance metrics.
    
        Attributes:
            total_time (float): The total time.
            total_predictions (int): The total number of predictions.
            name (Optional[str]): The name of the corresponding experiment.
        
        Properties:
            time_per_prediction (float): The time per prediction.
    """
    total_time: float
    total_predictions: int
    name: typing.Optional[str] = None

    @property
    def time_per_prediction(self) -> float:
        """ The time per prediction."""
        return self.total_time / self.total_predictions

    def __iter__(self):
        """ Iterates over the dataclass, including computed properties."""
        all_attributes_dict = self.__dict__
        all_attributes_dict['time_per_prediction'] = self.time_per_prediction
        for key, value in all_attributes_dict.items():
            yield key, value


def time_predictions(model: ImplementsPredict,
                     data_to_predict: Dataset,
                     name: typing.Optional[str] = None) -> Performance:
    """ Timer for how long a model takes to make predictions on Performance.

        Args:
            model (tf.keras.Model): The model to run performance metrics on.
            data_to_predict (Dataset): The data to make predictions on.
            name (Optional[str]): The name of the corresponding experiment.

        Returns:
            (Performance) The model performance.        
    """
    start_time = time.perf_counter()
    _ = model.predict(data_to_predict)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    return Performance(total_time=total_time, total_predictions=len(data_to_predict), name=name)


def generate_performance_dataframe(performance_metrics: typing.List[Performance]) -> pd.DataFrame:
    """ Generates a dataframe from the performance metrics.

        Args:
            performance_metrics (List[Performance]): The performance metrics.

        Returns:
            (pd.DataFrame) The dataframe of all the performance metrics.
    """
    performance_dict = {}
    for index, metric in enumerate(performance_metrics):
        name = metric.name or f'experiment_{index}'
        performance_dict[name] = metric.time_per_prediction

    return pd.DataFrame(performance_dict).transpose()
