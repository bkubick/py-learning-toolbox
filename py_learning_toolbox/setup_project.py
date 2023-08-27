#!/usr/bin/env python3
# coding: utf-8
""" This module contains functions for setting up a new project. This includes functions for
    creating the directory structure for a new project.
"""

from __future__ import annotations

import os
import sys


def create_project_directories(project_directory: str) -> None:
    """ This function creates the directory structure for a new project.

    Args:
        project_directory (str): The path to the project directory.
    """
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)

    # Create the directory structure for the project
    directories = ['checkpoints', 'data', 'logs', 'models', 'notebooks']

    for directory in directories:
        new_directory = f'{project_directory}/{directory}'
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)


def main():
    """ This function is the main function for the module. It is used to create the directory
        structure for a new project.

        Raises:
            ValueError: If the project directory name or path is not provided.
    """
    if len(sys.argv) != 2:
        raise ValueError('Please provide the project directory name or path')

    project_directory = sys.argv[1]
    create_project_directories(project_directory)


if __name__ == '__main__':
    main()
