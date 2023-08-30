# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
import typing

from sklearn.model_selection import train_test_split


__all__ = ['Dataset', 'get_split_dataset_from_data']


class SeparatedData(typing.NamedTuple):
    """ A class for storing separated data.

        Attributes:
            X: The features.
            y: The labels.
    """
    X: typing.Any
    y: typing.Any


@dataclass(frozen=True)
class Dataset:
    """ A class for storing data.
    
        Attributes:
            train: The training data.
            val: The validation data.
            test: The testing data.
    """
    train: typing.Optional[typing.Union[SeparatedData, typing.Any]]
    test: typing.Optional[typing.Union[SeparatedData, typing.Any]] = None
    val: typing.Optional[typing.Union[SeparatedData, typing.Any]] = None

    @property
    def train_size(self) -> int:
        """ Returns the size of the training data."""
        return len(self.train)

    @property
    def test_size(self) -> int:
        """ Returns the size of the testing data.

            Raises:
                ValueError: If the testing data is None.
        """
        if self.test is None:
            raise ValueError('test is None. Please provide a testing dataset.')

        return len(self.test)

    @property
    def val_size(self) -> int:
        """ Returns the size of the validation data.
        
            Raises:
                ValueError: If the validation data is None.
        """
        if self.val is None:
            raise ValueError('val is None. Please provide a validation dataset.')

        return len(self.val)

    @property
    def data_type(self) -> typing.Any:
        """ Returns the data type of the dataset. """
        return type(self.train)


def get_split_dataset_from_data(X_data: typing.Any,
                                y_data: typing.Any,
                                test_size: float = 0.2,
                                shuffle: bool = True,
                                random_state: typing.Optional[int] = None) -> Dataset:
    """ Splits the data into training and testing data.

        Args:
            X_data (Any): The data to split.
            y_data (Any): The labels to split.
            test_size (float): The size of the testing data.
            shuffle (bool): Whether to shuffle the data.
            random_state (Optional[int]): The random state.

        Returns:
            (Dataset) The dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_data,
        y_data,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state)

    return Dataset(
        train=SeparatedData(X=X_train, y=y_train),
        test=SeparatedData(X=X_test, y=y_test))


def get_dataset_from_train_test_val(train_data: typing.Any, test_data: typing.Any, val_data: typing.Any) -> Dataset:
    """ Gets a dataset from the training, testing, and validation data.

        Args:
            train_data (Any): The training data.
            test_data (Any): The testing data.
            val_data (Any): The validation data.

        Raises:
            AssertionError: If the data types of the training, testing, and validation data are not the same.

        Returns:
            (Dataset) The dataset.
    """
    assert type(train_data) == type(test_data) == type(val_data), 'train, test, and val must be of the same type'

    return Dataset(
        train=train_data,
        test=test_data,
        val=val_data)
