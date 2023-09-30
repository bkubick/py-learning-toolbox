# coding: utf-8

from __future__ import annotations

from collections import Counter
import datetime as dt
import io
import os
import typing

import matplotlib.pyplot as plt
import tensorflow as tf


if typing.TYPE_CHECKING:
    import numpy as np

    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = ['export_embedding_projector_data', 'get_word_counts', 'plot_words_counts']


def export_embedding_projector_data(model_name: str,
                                    embedded_layer_weights: tf.Tensor,
                                    vocabulary: typing.List[str],
                                    filepath: typing.Optional[str] = None,
                                    include_timestamp: bool = False):
    """ Exports the embedding projector data to filenames listed below.
        - vectors.tsv : The embedding vectors.
        - metadata.tsv : The words in the vocabulary.

        Files are stored at:
            {filepath}/{model_name}/embedding_projector_data/{timestamp}

        This code, as well as the embedding projector, is based on the following resources:
        - https://www.tensorflow.org/text/guide/word_embeddings : See the section on "Retrieving the trained
          embeddings and their metadata".
        - https://projector.tensorflow.org/ : The embedding projector.

        Args:
            model_name (str): The name of the model.
            embedded_layer_weights (tf.Tensor): The weights of the embedded layer.
            vocabulary (typing.List[str]): The words in the vocabulary.
            filepath (str): The filepath to save the data to.
                Defaults to 'logs'
            include_timestamp (bool): Whether to include a timestamp in the filepath.
    """
    # Create embedding files (These will be uploaded to the embedding projector)
    embedding_data_filepath = f'{filepath or "logs"}/{model_name}/embedding_projector_data'

    if include_timestamp:
        embedding_data_filepath = f'{embedding_data_filepath}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    if not os.path.exists(embedding_data_filepath):
        os.makedirs(embedding_data_filepath)

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


def get_word_counts(sentences: ArrayLike) -> typing.Dict[str, int]:
    """ Gets the words by count in all the sentences.

        NOTE: This does not clean the sentences (capitalization and punctuation will result in
        different counts for the same word). If you want to clean the sentences,
        such that only words are included, you can use the `clean_text` function
        from `py_learning_toolbox/ml_toolbox/preprocessing/language.py`.
            
        Args:
            sentences (ArrayLike[str]): The sentences to get the words by count from.

        Returns:
            (Dict[str, int]) The words by count.
    """
    np_sentences = tf.strings.as_string(sentences).numpy()
    all_sentences = str(b' '.join(np_sentences), encoding='utf-8')

    return dict(Counter(all_sentences.split()))


def plot_words_counts(word_counts: typing.Dict[str, int],
                      n: int = 20,
                      most_common: bool = True,
                      figsize: typing.Tuple[int, int] = (8, 5),
                      tick_fontsize: int = 8) -> None:
    """ Plots the n most or least common words based on the counts.

        Args:
            word_counts (Dict[str, int]): the word count by each word (i.e. [{'obi-wan': 4}... ]).
                NOTE: a complimentary function to generate this is `get_word_counts`.
            n (int): the number of words to include in plot.
            most_common (bool): whether to plot the most or least common words.
            tick_fontsize (int): the fontsize for the x and y ticks.
    """
    sorted_words = sorted(word_counts, key=lambda x: word_counts[x])
    counts = [word_counts[word] for word in sorted_words]

    n_words = sorted_words[-n:] if most_common else sorted_words[:n]
    n_counts = counts[-n:] if most_common else counts[:n]

    title = f'Top {n} {"Most" if most_common else "Least"} Commonly Used Words'

    plt.figure(figsize=figsize)
    plt.bar(n_words, n_counts)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    plt.yticks(rotation=0, fontsize=tick_fontsize)
    plt.xlabel('Most Common Words:', fontsize=12)
    plt.ylabel('Number of Occurences:', fontsize=12)
    plt.title(title, fontsize=12)
