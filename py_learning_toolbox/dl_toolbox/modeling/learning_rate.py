# coding: utf-8

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    Number = typing.Union[int, float]
    CallbackFunction = typing.Callable[[int], float]


def generate_exponential_decay_function(
        initial_learning_rate: Number = 1e-3,
        decay_factor: Number = 0.1,
        decay_step: Number = 10,
        warmup_epochs: int = 0,
        min_learning_rate: typing.Optional[Number] = None,
        max_learning_rate: typing.Optional[Number] = None) -> CallbackFunction:
    """ Returns a function that decays the learning rate exponentially for each epoch.

        NOTE: This is a function commonly used with the LearningRateScheduler callback.
    
        Args:
            initial_learning_rate (float): The initial learning rate.
            decay_factor (Number): The decay factor.
            decay_step (Number): The decay step.
            warmup_epochs (int): The number of epochs to warm up on before adjusting the learning rate.
            min_learning_rate (Optional[Number]): The minimum learning rate allowed.
            max_learning_rate (Optional[Number]): The maximum learning rate allowed.

        Returns:
            A function that decays the learning rate exponentially.
    """
    def _exponential_decay(epoch: int, *args, **kwargs) -> float:
        if epoch < warmup_epochs:
            new_learning_rate = initial_learning_rate
        else:
            new_learning_rate = initial_learning_rate * decay_factor ** (epoch / decay_step)

        if min_learning_rate is not None and new_learning_rate < min_learning_rate:
            return min_learning_rate
        elif max_learning_rate is not None and new_learning_rate > max_learning_rate:
            return max_learning_rate
        else:
            return new_learning_rate
        
    return _exponential_decay
