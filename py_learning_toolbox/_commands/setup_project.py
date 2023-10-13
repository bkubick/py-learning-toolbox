#!/usr/bin/env python3
# coding: utf-8
""" This module contains functions for setting up a new project. This includes functions for
    creating the directory structure for a new project.
"""

from __future__ import annotations

import json
import os


def create_project_directories(project_directory: str) -> None:
    """ This function creates the directory structure for a new project.

        The following directory structure is created:

        project_directory
        ├── checkpoints
        ├── data
        ├── logs
        └── models

        Args:
            project_directory (str): The path to the project directory.
    """
    if not os.path.exists(project_directory):
        os.makedirs(project_directory)

    # Create the directory structure for the project
    directories = ['checkpoints', 'data', 'logs', 'models']

    for directory in directories:
        new_directory = f'{project_directory}/{directory}'
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)


def create_notebook_template(notebook_path: str, notebook_name: str) -> None:
    """ This function creates a template for a Jupyter notebook.

        DEV NOTE: the notebook template is stored in the `templates` directory.
        If you change the template, ensure this function is updated accordingly.
    
        Args:
            notebook_path (str): The path to the notebook.
            notebook_name (str): The name of the notebook.
    """
    if not os.path.exists(notebook_path):
        os.makedirs(notebook_path)

    if not notebook_name.endswith('.ipynb'):
        notebook_name += '.ipynb'

    notebook_path = f'{notebook_path}/{notebook_name}'

    if os.path.exists(notebook_path):
        print(f'{notebook_path} already exists')
        return

    path = '/'.join(__file__.split('/')[:-1])
    notebook_title = notebook_name.replace('.ipynb', '').replace('_', ' ').title()
    notebook_template = json.load(open(f'{path}/templates/notebook.json'))
    notebook_template['cells'][0]['source'][0] = notebook_template['cells'][0]['source'][0].replace('PROJECT_TITLE', notebook_title)

    with open(notebook_path, 'w') as f:
        json.dump(notebook_template, f)
