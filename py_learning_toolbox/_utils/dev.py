# coding: utf-8

from __future__ import annotations

import logging
import typing


__all__ = ['obsolete']


def obsolete(message: typing.Optional[str] = None) -> typing.Callable:
    """ Decorator to mark a function as obsolete.

        Args:
            message (Optional[str]): the message to log when the function is called.

        Returns:
            (typing.Callable) the wrapped function.
    """

    def function_wrapper(func: typing.Callable) -> typing.Callable:
        """ Decorator to mark a function as deprecated.

            Args:
                func (typing.Callable): the function to mark as deprecated.

            Returns:
                (typing.Callable) the wrapped function.
        """
        def logger(*args, **kwargs):
            nonlocal message
            message = message or f'{func.__name__} is obsolete'
            logging.warning(message)

            return func(*args, **kwargs)

        return logger
    
    return function_wrapper
