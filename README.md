# PyLearningToolbox (py-learning-toolbox)

## Description
All-in-One intelligence toolbox yielding my personal tools I commonly use in AI, Machine Learning, Deep Learning, and Genetic Algorithm projects.

**IMPORTANT NOTE** This repository is under active development, and any versioning will be inconsistent until the repository becomes more stable. Once it becomes more stable, I will begin tracking the versioning.

## Tech Stack
<img style="padding-right:20px;" align=left alt="TensorFlow" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
<img style="padding-right:20px;" alt="TensorFlow" align=left src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/>
<img style="padding-right:20px;" alt="Matplotlib" align=left src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)"/>
<img style="padding-right:20px;" alt="Numpy" align=left src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
<img style="padding-right:20px;" alt="Pandas" align=left src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"/>
<img style="padding-right:20px;" alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"/>

## Installing
This package is not setup to install directly from pypi. To install this package to an environment, run the following command.

```
pip install py-learning-toolbox@git+https://github.com/bkubick/py-learning-toolbox.git
```

## Importing
This package can only be used by importing submodules. This cannot be used by importing the entire `py_learning_toolbox` library to limit performance requirements on some of the utilities. For instance, if you want to utilize the `ml_toolbox` module utilities, you can run the following import statement to access all the deep learning tools through the `ml_toolbox` namespace.

```
from py_learning_toolbox import ml_toolbox
```

## Commands

This repo relies on a specific structure when handling files for a given project, and offers a handful of commands to make making new projects significantly easier.

### Project Setup
Due to some of the default directory names I use as the path to store logs, checkpoints, and models, there is a desired directory setup for each project that is preferred when using these utilities. To properly setup the directory structure, after the package is installed on local, run the following command to create the directory structure for each corresponding project being worked on.

Additionally, the python notebook layout for many projects were just copied over at the beginning of each project, so to mitigate the amount of time used to create new notebooks for projects, the command also has the option to create a jupyter notebook with the same name as the project name.

```
pltb create_project [project_name_or_dir]
```

To include the notebook creation:

```
pltb create_project [project_name_or_dir] --notebook
```

For instance, if you want to look at tweets to determine if it is talking about a disaster or not, you might want to setup a project called `disaster_tweets` and it lives within the `projects` directory (`projects/disaster_tweets`). To do this, the command would look like:

```
pltb create_project projects/disaster_tweets --notebook
```

Running this command creates the required dirs in `projects/disaster_tweets`, as well as the `disaster_tweets.ipynb` that lives in the directory.


### Notebook Setup
If you already created the project without a notebook, but want to generate an `.ipynb` for the project, run the following command.

```
pltb create_notebook [project_dir] [notebook_name]
```

## Additional Work
This repository is under active development that I want to dig into deeper as I continue to grow in the field of deep learning. See below for the additional work I plan to work on.

1. Cleanup typehinting for the dataframe, arraylike, etc. Make utilities handle the arraylike type based off the various types.
2. Read through various machine learning papers to determine ideal metrics, plots to include, etc.
3. Make this compatible with both TensorFlow and PyTorch.
4. Add unit tests.
