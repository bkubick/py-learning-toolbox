# coding: utf-8

from __future__ import annotations

import datetime as dt
import io
import typing

if typing.TYPE_CHECKING:
    import tensorflow as tf


__all__ = ['export_embedding_projector_data']


def export_embedding_projector_data(embedded_layer_weights: tf.Tensor,
                                    vocabulary: typing.List[str],
                                    filepath: typing.Optional[str] = None,
                                    include_timestamp: bool = False):
    """ Exports the embedding projector data to filenames listed below.
        - vectors.tsv : The embedding vectors.
        - metadata.tsv : The words in the vocabulary.

        This code, as well as the embedding projector, is based on the following resources:
        - https://www.tensorflow.org/text/guide/word_embeddings : See the section on "Retrieving the trained
          embeddings and their metadata".
        - https://projector.tensorflow.org/ : The embedding projector.

        Args:
            embedded_layer_weights (tf.Tensor): The weights of the embedded layer.
            vocabulary (typing.List[str]): The words in the vocabulary.
            filepath (str): The filepath to save the data to.
            include_timestamp (bool): Whether to include a timestamp in the filepath.
    """
    # Create embedding files (These will be uploaded to the embedding projector)
    embedding_data_filepath = filepath or '.'

    if include_timestamp:
        embedding_data_filepath = f'{embedding_data_filepath}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    out_v = io.open(f'{embedding_data_filepath}/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open(f'{embedding_data_filepath}/metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocabulary):
        if index == 0:
            continue  # skip 0, it's padding.

        vec = embedded_layer_weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")

    out_v.close()
    out_m.close()
