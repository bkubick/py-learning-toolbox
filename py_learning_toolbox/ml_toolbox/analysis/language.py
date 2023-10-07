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
    from sklearn.preprocessing import OneHotEncoder

    ArrayLike = typing.Union[tf.Tensor, typing.List[typing.Any], np.ndarray]


__all__ = [
    'export_embedding_projector_data',
    'generate_text_from_seed',
    'get_character_counts',
    'get_word_counts',
    'plot_token_counts',
]


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


def get_character_counts(sentences: ArrayLike) -> typing.Dict[str, int]:
    """ Gets the characters by count in all the sentences.

        NOTE: This does not clean the sentences (capitalization and punctuation will result in
        different counts for the same character). If you want to clean the sentences,
        such that only characters are included, you can use the `clean_text` function
        from `py_learning_toolbox/ml_toolbox/preprocessing/language.py`.
            
        Args:
            sentences (ArrayLike[str]): The sentences to get the characters by count from.

        Returns:
            (Dict[str, int]) The characters by count.
    """
    np_sentences = tf.strings.as_string(sentences).numpy()
    all_sentences = str(b' '.join(np_sentences), encoding='utf-8')

    return dict(Counter(all_sentences))


def plot_token_counts(token_counts: typing.Dict[str, int],
                      n: int = 20,
                      most_common: bool = True,
                      figsize: typing.Tuple[int, int] = (8, 5),
                      tick_fontsize: int = 8) -> None:
    """ Plots the n most or least common tokens (characters, words, etc.) based on the counts.

        Args:
            token_counts (Dict[str, int]): the token count by each word (i.e. [{'obi-wan': 4}... ]).
                NOTE: a complimentary function to generate this is `get_word_counts`.
            n (int): the number of tokens to include in plot.
            most_common (bool): whether to plot the most or least common tokens.
            tick_fontsize (int): the fontsize for the x and y ticks.
    """
    sorted_tokens = sorted(token_counts, key=lambda x: token_counts[x])
    counts = [token_counts[token] for token in sorted_tokens]

    n_tokens = sorted_tokens[-n:] if most_common else sorted_tokens[:n]
    n_counts = counts[-n:] if most_common else counts[:n]

    title = f'Top {n} {"Most" if most_common else "Least"} Commonly Used Tokens'

    plt.figure(figsize=figsize)
    plt.bar(n_tokens, n_counts)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    plt.yticks(rotation=0, fontsize=tick_fontsize)
    plt.xlabel('Most Common Tokens', fontsize=12)
    plt.ylabel('Number of Occurences', fontsize=12)
    plt.title(title, fontsize=12)


def generate_text_from_seed(seed_text: str,
                            num_preds: int,
                            model: tf.keras.models.Model,
                            one_hot: OneHotEncoder,
                            pred_type: typing.Optional[str] = None) -> str:
    """ Generates a sentence from the seed text with the designated model and encoder mapper.

        Args:
            seed_text (str): the starting text for the sentence to be generated.
            num_preds (int): how many predictions (chars/words) to add.
            model (Model): the model to use to generate the text.
            one_hot (OneHotEncoder): the encoder used to convert numerical indices to text.
            pred_type (Optional[str]): the prediction type. Either 'char' or 'word'.
                Defaults to 'word'.
        
        Raises:
            ValueError: if pred_type is not either 'char' or 'word'.

        Returns:
            (str) the generated text.
    """
    pred_type = pred_type or 'word'
    if pred_type not in {'char', 'word'}:
        raise ValueError('pred_type must be either "char" or "word"')

    generated_text = seed_text
    for _ in range(num_preds):
        pred_probs = model.predict([generated_text], verbose=0)
        pred_index = tf.argmax(pred_probs, axis=1)[0]
        # Due to the size of the list, rounding doesn't always gurantee a selected word
        pred_probs[0][pred_index] = 1
        pred_vector = tf.round(pred_probs)
        predicted_item = str(one_hot.inverse_transform(pred_vector)[0][0])

        generated_text += f'{predicted_item}'  if pred_type == 'char' else f' {predicted_item}'

    return generated_text
