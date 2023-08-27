# PyLearningToolbox (py-learning-toolbox)

## Description
All-in-One intelligence toolbox yielding my personal tools I commonly use in AI, Machine Learning, Deep Learning, and Genetic Algorithm projects.

## Installing
This package is not setup to install directly from pypi. To install this package to an environment, run the following command.

```
pip install py-learning-toolbox@git+https://github.com/bkubick/py-learning-toolbox.git
```

## Importing
This package can only be used by importing submodules. This cannot be used by importing the entire `py_learning_toolbox` library to limit performance requirements on some of the utilities. For instance, if you want to utilize the `deep_learning` module utilities, you can run the following import statement to access all the deep learning tools through the `deep_learning` namespace.

```
from py_learning_toolbox import deep_learning
```
