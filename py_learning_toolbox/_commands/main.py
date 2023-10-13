# coding: utf-8

from __future__ import annotations

import sys
import typing

from .setup_project import create_notebook_template, create_project_directories


HELP_ALIAS = {'--h', '--help'}


def help() -> None:
    """ This function prints the help message for the `pltb` commands.
    """
    print('Commands:')
    print('    --help: Print the help message.')
    print('    create_project: Create the directory structure for a new project.')
    print('    create_notebook: Create a template for a Jupyter notebook.')
    print('    create_dirs: Create the directory structure for a new project.')


def create_project() -> None:
    """ This function creates the directory structure for a new project.
    """
    create_dirs()
    create_notebook()


def create_dirs() -> None:
    """ This function creates the directory structure for a new project.
    """
    if len(sys.argv) < 3:
        print('Please provide the project directory name or path or use `pltb create_dirs --help` for more information')
        return

    if sys.argv[2] in HELP_ALIAS:
        print('Creates the directory structure for a new project.')
        print('    Usage: pltb create_dirs <project_directory>')
        return

    create_project_directories(sys.argv[2])


def create_notebook() -> None:
    """ This function creates a template for a Jupyter notebook.
    """
    if len(sys.argv) < 3:
        print('Please provide the notebook path and name or use `pltb create_notebook --help` for more information')
        return

    if sys.argv[2] in HELP_ALIAS:
        print('Creates a template for a Jupyter notebook.')
        print('    Usage: pltb create_notebook <notebook_path> <notebook_name>')
        return

    filepath = sys.argv[2]
    filename = sys.argv[3] if len(sys.argv) > 3 else filepath.split('/')[-1]

    create_notebook_template(filepath, filename)


def main():
    """ This function is the entry point for the `pltb` commands. It is used to call the
        appropriate function based on the command provided.

        Commands:
            --help: Print the help message (--h, --help).
            create_project: Create the directory structure for a new project.
            create_notebook: Create a template for a Jupyter notebook.
            create_dirs: Create the directory structure for a new project.
    """
    if len(sys.argv) == 1:
        print('Please provide the command name')
        return

    commands = {
        'create_project': create_project,
        'create_notebook': create_notebook,
        'create_dirs': create_dirs,
    }

    for help_command in HELP_ALIAS:
        commands[help_command] = help

    command: typing.Optional[typing.Callable] = commands.get(sys.argv[1])
    if command is None:
        print(f'Invalid command: {sys.argv[1]}')
        return
    else:
        command()


if __name__ == '__main__':
    main()
