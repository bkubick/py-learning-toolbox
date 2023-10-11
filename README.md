# PyLearningToolbox (py-learning-toolbox)

## Description
All-in-One intelligence toolbox yielding my personal tools I commonly use in AI, Machine Learning, Deep Learning, and Genetic Algorithm projects.

**IMPORTANT NOTE** This repository is under active development, and any versioning will be inconsistent until the repository becomes more stable. Once it becomes more stable, I will begin tracking the versioning.

## Tech Stack
<img style="padding-right:20px;" align=left alt="TensorFlow" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
<img style="padding-right:20px;" alt="TensorFlow" align=left src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/>
<img style="padding-right:20px;" alt="Matplotlib" align=left src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)"/>
<img style="padding-right:20px;" alt="Numpy" align=left src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
<img style="padding-right:20px;" alt="Pandas" src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"/>

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

## Additional Work
This repository is under active development that I want to dig into deeper as I continue to grow in the field of deep learning. See below for the additional work I plan to work on.

1. Cleanup typehinting for the dataframe, arraylike, etc.
2. Review my `deep-learning-development` repository, and add over any utilities I think would be beneficial to live here.
3. Read through various machine learning papers to determine ideal metrics, plots to include, etc.
4. Make this compatible with both TensorFlow and PyTorch.
