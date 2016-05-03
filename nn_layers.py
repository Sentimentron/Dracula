"""
    This file contains the rest of the layers in the network.
"""

import theano
from theano import tensor
from util import numpy_floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import tensorflow as tf

def embeddings_layer(x, Wemb):
    """
    Returns the one-hot vector x after encoding in the Wemb embedding space.
    :param x: One-hot index vector (25, 23...)
    :param Wemb: Word embeddings
    :return:
    """

    return tf.gather(Wemb, x)

def lstm_mask_layer(proj, mask):
    """
    Removes any spurious output from the LSTM that's not covered by a label
    or doesn't correspond to any real input.
    :param proj: Output of the LSTM layer
    :param mask: 1 if the position is valid, 0 otherwise
    :return: The masked values
    """

    return tf.matmul(proj, mask)

def per_word_averaging_layer(dist, dist_mask):
    """
    Average everything per-word.
    :param proj: Output of the LSTM layer
    :param wmask: The map of word -> character embeddings as a [word, character_index, tweet, all_ones_dim_proj] matrix
    :param maxw: The maximum character-per-word offset
    :param trim: Whether to trim undefined regions in the result
    :return: The per-word averages
    """

    dist = tf.mul(dist, dist_mask)

    # Transpose everything so it's the same as Theano
    dist = tf.transpose(dist, [1, 2, 0, 3])

    dist_mask = tf.transpose(dist_mask, [1, 2, 0, 3])

    divider = tf.reduce_sum(dist_mask, 0)
    divider = tf.cast(divider, dtype='float32')
    normalizer = tf.equal(divider, 0.0)
    normalizer = tf.cast(normalizer, dtype='float32')
    divider = tf.add(divider, normalizer)

    tmp = tf.reduce_sum(dist, 0)
    tmp = tf.div(tmp, divider)

    return tf.transpose(tmp, [1, 0, 2])

def logits_layer(avg_per_word, U, b, y_mask, maxw, training=False):
    """
    Unscaled logits output.
    :param avg_per_word: Output from word-averaging
    :param U: Classification weight matrix
    :param b: Classification bias layer
    :param y_mask: Because not all fragments are the same length, set y_mask to 0 in those positions
                    where the output is undefined, causing this thing to output the special 0 label (for "don't care")
    :return: Softmax predictions
    """
    preds, raw_preds = None, None

    def softmax_fn(current_input):
        return tf.matmul(current_input, U) + b

    #avg_per_word = tf.Print(avg_per_word, [avg_per_word], message="avg")
    if training:
        keep_prob = 0.5
        dropout_mask = tf.nn.dropout(avg_per_word, keep_prob)
        raw_preds = tf.map_fn(softmax_fn, dropout_mask)
    else:
        raw_preds = tf.map_fn(softmax_fn, avg_per_word)

#    return tf.mul(raw_preds, y_mask)
    return raw_preds

