# coding: utf-8
""" Custom types used throughout the toolbox.

    NOTE: This module is private and should not be imported directly. Instead, import the public modules.

    Types:
        TensorLike: A type that can be used to represent a tensor like object. This includes tf.Tensor, np.ndarray,
"""

from __future__ import annotations

import typing

import numpy as np
import tensorflow as tf


__all__ = ['TensorLike']


TensorLike = typing.Union[tf.Tensor, typing.List, np.ndarray]
