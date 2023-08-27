# coding: utf-8
import logging

logger = logging.getLogger(__name__)


def export_file_to_local(filename: str) -> bool:
    """ Exports a file to the local machine.

        Args:
            filename (str): The filename to export from google colab to local.

        Returns:
            bool: True if the file was exported, False otherwise.
    """
    try:
        from google.colab import files
        files.download(filename)

        return True
    except Exception:
        logger.info(f'Could not export the following file to local: {filename}')
        return False
