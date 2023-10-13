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


def create_project() -> None:
    """ This function creates the directory structure for a new project.
    """
    if len(sys.argv) < 3:
        print('Please provide the project directory name or path or use `pltb create_project --help` for more information')
        return

    if sys.argv[2] in HELP_ALIAS:
        print('Creates the directory structure for a new project.')
        print('    Usage: pltb create_project <project_directory>')
        return

    create_project_directories(sys.argv[2])

    if len(sys.argv) >= 4 and sys.argv[3] == '--notebook':
        create_notebook()


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

    if len(sys.argv) >= 4 and sys.argv[3] == '--notebook':
        filename = filepath.split('/')[-1]
    elif len(sys.argv) >= 4:
        filename = sys.argv[3]
    else:
        filename = filepath.split('/')[-1]

    create_notebook_template(filepath, filename)


def main():
    """ This function is the entry point for the `pltb` commands. It is used to call the
        appropriate function based on the command provided.

        Commands:
            --help: Print the help message (--h, --help).
            create_project: Create the directory structure for a new project.
            create_notebook: Create a template for a Jupyter notebook.
    """
    if len(sys.argv) == 1:
        print('Please provide the command name')
        return

    commands = {
        'create_project': create_project,
        'create_notebook': create_notebook,
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
