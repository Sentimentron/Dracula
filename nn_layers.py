"""
    This file contains the rest of the layers in the network.
"""

import theano
from theano import tensor
from util import numpy_floatX

import tensorflow as tf

def flatten(tensor):
    pass

def embeddings_layer(x, Wemb):
    """
    Returns the matrix x after encoding in the Wemb embedding space.
    :param x: Indice matrix (25, 23...)
    :return: A Tensor containing the values looked up
    """

    return tf.gather(Wemb, x)


def lstm_mask_layer(proj, mask):
    """
    Removes any spurioeus output from the LSTM that's not covered by a label
    or doesn't correspond to any real input.
    :param proj: Output of the LSTM layer
    :param mask: 1 if the position is valid, 0 otherwise
    :return: The masked values
    """

    return tf.matmul(proj, mask)
    #return proj * mask[:, :, None]

def per_word_averaging_layer_distrib(proj, wmask, maxw):
    """
    Create the per-word embedding matrix
    :param proj: The LSTM-layer output from below
    :param wmask: The map of word -> character embeddings as a [word, character_index, tweet, all_ones_dim_proj] matrix
    :param maxw: The maximum word offset
    :return: A layer of projected word embeddings distributed the associated word indices in
        [word, character_index, tweet, all_ones_dim_proj]
    """

    # Duplicate and expand the input matrices by the required number of characters
    dup = [tf.expand_dims(proj, 0) for _ in range(maxw)]
    dup = tf.concat(0, dup)

    # Element-wise multiply to obtain the final result
    mul = tf.mul(wmask, proj)

    return mul

def per_word_averaging_layer(proj, wmask, maxw, trim=True):
    """
    Average everything per-word
    :param proj: Output of the LSTM layer
    :param wmask: The map of word -> character embeddings as a [word, character_index, tweet, all_ones_dim_proj] matrix
    :param maxw: The maximum character-per-word offset
    :param trim: Whether to trim undefined regions in the result
    :return: The per-word averages
    """

    dist = per_word_averaging_layer_distrib(proj, wmask, maxw)

    # Transpose the result so the order's the same as Theano
    dist = tf.transpose(dist, [1, 2, 0, 3])

    # Count the number of characters inside each word
    divider = tf.not_equal(dist, tf.constant(0.), name='divider')
    divider = tf.cast(divider, dtype='float32')
    divider = tf.reduce_sum(divider, 0)

    # Normalize so we don't get NaNs (minimum value should be 1.)
    normalizer = tf.equal(divider, tf.constant(0.))
    divider = tf.add(divider, tf.cast(normalizer, dtype='float32'))

    # Divide everything together
    tmp = tf.reduce_sum(dist, 0)
    tmp = tf.div(tmp, divider)

    if not trim:
        return tmp
    else:
        return tf.slice(tmp, begin=[0, 1, 0], size=[-1, -1, -1])

def softmax_layer(dropout_mask, avg_per_word, U, b, y_mask):
    """
    Produces the final labels via softmax
    :param avg_per_word: Output from word-averaging
    :param U: Classification weight matrix
    :param b: Classification bias layer
    :param y_mask: Because not all fragments are the same length, set y_mask to 0 in those positions
                    where the output is undefined, causing this thing to output the special 0 label (for "don't care")
    :return: Softmax predictions
    """
    #avg_per_word = theano.printing.Print("avg_per_word")(avg_per_word)
    raw_pred, _ = theano.scan(fn=lambda p, free_variable: tensor.nnet.softmax(tensor.dot(p, U * dropout_mask) + b),
                              outputs_info=None,
                              sequences=[avg_per_word, tensor.arange(16)]
                              )

    #raw_pred = theano.tensor.printing.Print("raw_pred")(raw_pred)
    #y_mask = theano.tensor.printing.Print("y_mask")(y_mask)
    pred = tensor.zeros_like(raw_pred)
    pred = tensor.inc_subtensor(pred[:, :, 0], 1)
    pred = tensor.set_subtensor(pred[y_mask.nonzero()], raw_pred[y_mask.nonzero()])

    return pred
