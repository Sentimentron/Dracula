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
    :param proj: Output of the LSTM layer
    :param dist_mask: 4D index tensor
    :return: The per-word averages.
    """

    dist = dist * dist_mask

    dist = dist.dimshuffle(1, 2, 0, 3)
    dist_mask = tensor.cast(dist_mask, theano.config.floatX)
    dist_mask = dist_mask.dimshuffle(1, 2, 0, 3)
    divider = dist_mask.sum(axis=0)
    divider += tensor.eq(divider, numpy_floatX(0.0))

    dist = dist * dist_mask
    tmp = tensor.cast(dist.sum(axis=0), theano.config.floatX)
    tmp /= divider
    return tmp.dimshuffle(1, 0, 2)

def softmax_layer(avg_per_word, U, b, y_mask, maxw, training=False):
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
    if training:
        srng = RandomStreams(seed=12345)
        dropout_mask = tensor.cast(srng.binomial(size=U.shape, p=0.5), theano.config.floatX)
        #U = theano.printing.Print("U", attrs=["shape"])(U)
        #dropout_mask = theano.printing.Print("dropout_mask", attrs=["shape"])(dropout_mask)
        raw_pred, _ = theano.scan(fn=lambda p, free_variable: tensor.nnet.softmax(tensor.dot(p, tensor.mul(U, dropout_mask)) + b),
                                  outputs_info=None,
                                  sequences=[avg_per_word, tensor.arange(maxw)]
                                  )
    else:
        raw_pred, _ = theano.scan(fn=lambda p, free_variable: tensor.nnet.softmax(tensor.dot(p, U) + b),
				  outputs_info=None,
				  sequences=[avg_per_word, tensor.arange(maxw)]
				  )

    #raw_pred = theano.tensor.printing.Print("raw_pred")(raw_pred)
    #y_mask = theano.tensor.printing.Print("y_mask")(y_mask)
    pred = tensor.zeros_like(raw_pred)
    pred = tensor.inc_subtensor(pred[:, :, 0], 1)
    pred = tensor.set_subtensor(pred[y_mask.nonzero()], raw_pred[y_mask.nonzero()])

    return pred
