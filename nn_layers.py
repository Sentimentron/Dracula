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

def softmax_layer(avg_per_word, U, b, y_mask, maxw, training=False, scale=True):
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

    def softmax(x):
        e_x = tf.exp(x - tf.expand_dims(tf.reduce_max(x, 1), -1))
        return e_x / tf.expand_dims(tf.reduce_sum(e_x, 1), -1)

    def softmax_fn(current_input):
        logits = tf.matmul(current_input, U) + b
        if scale:
            return tf.nn.softmax(logits)
            #return softmax(logits)
        return logits

        #return softmax(tf.matmul(current_input, U) + b)

    #avg_per_word = tf.Print(avg_per_word, [avg_per_word], message="avg")
    if training:
        keep_prob = 0.50
        dropout_mask = tf.nn.dropout(avg_per_word, keep_prob)
        raw_preds = tf.map_fn(softmax_fn, dropout_mask)
    else:
        raw_preds = tf.map_fn(softmax_fn, avg_per_word)

    return raw_preds

    if True:

        #dupd_mask = tf.tile(y_mask, [1, 1, 46]) # TODO: don't hard-code ydim
        expanded_y_mask = tf.expand_dims(y_mask, -1)
        inverse_mask = 1 - expanded_y_mask # 1 in places where y_mask is zero
        dupd_mask = tf.concat(2, [expanded_y_mask for _ in range(46)]) # TODO: don't hard-code ydim
        new_mask = tf.concat(2, [inverse_mask, dupd_mask])

        expanded_tiled_y_mask = tf.concat(2, [expanded_y_mask for _ in range(46)])

        default_pred_1 = tf.ones_like(expanded_y_mask)
        default_pred_0 = tf.zeros_like(expanded_y_mask)
        default_pred_0 = tf.concat(2, [default_pred_0 for _ in range(45)])
        default_pred = tf.concat(2, [default_pred_1, default_pred_0])

        #expander = tf.zeros_like(raw_preds)
        #expander[:, :, 0] = tf.ones(tf.shape(expander)[2])
        #preds = tf.cond(tf.equal(y_mask, 0), new_mask, raw_preds)
        preds = tf.select(tf.equal(expanded_tiled_y_mask, 0), default_pred, raw_preds)
        #expander = tf.zeros_like(raw_preds)
        #    mask = expander * y_mask
        #return tf.mul(raw_preds, y_mask)

        #return tf.mul(mask, raw_preds)
        preds = tf.Print(preds, [preds], "preds=")
        return preds

