# PyLearningToolbox (py-learning-toolbox)

## Description
All-in-One intelligence toolbox yielding my personal tools I commonly use in AI, Machine Learning, Deep Learning, and Genetic Algorithm projects.

## Installing
This package is not setup to install directly from pypi. To install this package to an environment, run the following command.

```
pip install py-learning-toolbox@git+https://github.com/bkubick/py-learning-toolbox.git
```

## Importing
This package can only be used by importing submodules. This cannot be used by importing the entire `py_learning_toolbox` library to limit performance requirements on some of the utilities. For instance, if you want to utilize the `dl_toolbox` module utilities, you can run the following import statement to access all the deep learning tools through the `dl_toolbox` namespace.

```
from py_learning_toolbox import dl_toolbox
```

## Project Setup
Due to some of the default directory names I use as the path to store logs, checkpoints, and models, there is a desired directory setup for each project that is preferred when using these utilities. To properly setup the directory structure, after the package is installed on local, run the following command to create the dirtectory structure for each corresponding project being worked on.

```
pltb_setup_project [project_name_or_dir]
```

For instance, if you want to look at tweets to determine if it is talking about a disaster or not, you might want to setup a project called `disaster_tweets` and it lives within the `projects` directory (`projects/disaster_tweets`). To do this, the command would look like:

```
pltb_setup_project projects/disaster_tweets
```
