# coding: utf-8

from __future__ import annotations

import datetime as dt
import io
import os
import typing

import matplotlib.pyplot as plt


if typing.TYPE_CHECKING:
    import tensorflow as tf


__all__ = ['export_embedding_projector_data']


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


def get_words_by_count(sentences: typing.List[str]) -> typing.Dict[str, int]:
    """ Gets the words by count in all the sentences.
            
        Args:
            sentences (List[str]): The sentences to get the words by count from.

        Returns:
            (Dict[str, int]) The words by count.
    """
    words_by_count = {}
    for sentence in sentences:
        if len(sentence) == 1:
            continue

        for word in sentence.split(' '):
            words_by_count[word] = words_by_count.get(word, 0) + 1

    return words_by_count


def plot_words_by_count(words_by_count: typing.Dict[str, int],
                        n: int = 20,
                        most_common: bool = True,
                        figsize: typing.Tuple[int, int] = (8, 5),
                        tick_fontsize: int = 8) -> None:
    """ Plots the n most or least common words based on the counts.

        Args:
            words_by_count (Dict[str, int]): the word count by each word (i.e. [{'obi-wan': 4}... ]).
                NOTE: a complimentary function to generate this is `get_words_by_count`.
            n (int): the number of words to include in plot.
            most_common (bool): whether to plot the most or least common words.
            tick_fontsize (int): the fontsize for the x and y ticks.
    """
    sorted_words = sorted(words_by_count, key=lambda x: words_by_count[x])
    counts = [words_by_count[word] for word in sorted_words]

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
