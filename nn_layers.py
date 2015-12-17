"""
    This file contains the rest of the layers in the network.
"""

import theano
from theano import tensor
from util import numpy_floatX


def embeddings_layer(x, Wemb, n_timesteps, n_samples, dim_proj):
    """
    Returns the one-hot vector x after encoding in the Wemb embedding space.
    :param x: One-hot index vector (25, 23...)
    :param Wemb: Word embeddings
    :param n_timesteps: Maximum number of timesteps
    :param n_samples: Maximum number of samples
    :param dim_proj: Size of the word embeddings space
    :return:
    """

    return Wemb[x.flatten()].reshape([n_timesteps, n_samples, dim_proj])


def lstm_mask_layer(proj, mask):
    """
    Removes any spurious output from the LSTM that's not covered by a label
    or doesn't correspond to any real input.
    :param proj: Output of the LSTM layer
    :param mask: 1 if the position is valid, 0 otherwise
    :return: The masked values
    """

    return proj * mask[:, :, None]

def per_word_averaging_layer_distrib(proj, wmask, maxw):
    """

    """
    dup = [tensor.shape_padaxis(proj, 0) for _ in range(maxw)]
    dup = tensor.concatenate(dup, 0)
    #dup = tensor.shape_padaxis(proj, 0)

    mul = tensor.mul(wmask, dup)

    return mul

def per_word_averaging_layer(proj, wmask, maxw, trim=False):
    """
    :param proj: Output of the LSTM layer
    :param wmask: Unravelled 4D-index tensor (represented in 2d)
    :return: The per-word averages.
    """
    n_chars = proj.shape[0]
    n_samples = proj.shape[1]
    n_proj = proj.shape[2]

    dist = per_word_averaging_layer_distrib(proj, wmask, maxw)

    dist = dist.dimshuffle(1, 2, 0, 3)

    divider = tensor.cast(tensor.neq(dist, numpy_floatX(0.0)).sum(axis=0), theano.config.floatX)
    divider += tensor.eq(divider, numpy_floatX(0.0)) # Filter NaNs

    tmp = tensor.cast(dist.sum(axis=0), theano.config.floatX)
    tmp /= divider

    if not trim:
        return tmp
    else:
        ret = tensor.zeros_like(tmp)
        ret = tensor.set_subtensor(ret[:, :-1], tmp[:, 1:])
        return tensor.cast(ret, theano.config.floatX)

def softmax_layer(dropout_mask, avg_per_word, U, b, y_mask, maxw):
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
                              sequences=[avg_per_word, tensor.arange(maxw)]
                              )

    #raw_pred = theano.tensor.printing.Print("raw_pred")(raw_pred)
    #y_mask = theano.tensor.printing.Print("y_mask")(y_mask)
    pred = tensor.zeros_like(raw_pred)
    pred = tensor.inc_subtensor(pred[:, :, 0], 1)
    pred = tensor.set_subtensor(pred[y_mask.nonzero()], raw_pred[y_mask.nonzero()])

    return pred
