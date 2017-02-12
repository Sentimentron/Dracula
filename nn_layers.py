"""
    This file contains the rest of the layers in the network.
"""

import theano
from theano import tensor
from util import numpy_floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def embeddings_layer(x, Wemb, dim_proj):
    """
    Returns the one-hot vector x after encoding in the Wemb embedding space.
    :param x: One-hot index vector (25, 23...)
    :param Wemb: Word embeddings
    :return:
    """

    n_words = x.shape[0]
    n_max_letters_in_word = x.shape[1]
    n_batch = x.shape[2]

    dist = Wemb[x.flatten()].reshape([n_words, n_max_letters_in_word, n_batch, dim_proj])
    return dist


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
    print maxw, "MAXW"
    dup = [tensor.shape_padaxis(proj, 0) for _ in range(maxw)]
    dup = tensor.concatenate(dup, 0)
    #dup = tensor.shape_padaxis(proj, 0)

    mul = tensor.mul(wmask, dup)
    mul = theano.printing.Print("mul", attrs=["shape"])(mul)
#    mul = mul[mul.nonzero()]
#    mul = mul[mul != 0]
    compare = tensor.eq(mul, numpy_floatX(0.))
    mul = mul[(1-compare).nonzero()[0]]
    mul = theano.printing.Print("mul", attrs=["shape"])(mul)
#    mul = theano.printing.Print("mul")(mul)
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

    # tmp = theano.printing.Print("tmp", attrs=["shape"])(tmp)

    #_max = dist.max(axis=0)
    #_min = dist.min(axis=0)

    #tmp = tensor.concatenate([tmp, _max, _min], axis=2)
    #    tmp = theano.printing.Print("tmp", attrs=["shape"])(tmp)

    if not trim:
        return tmp
    else:
        ret = tensor.zeros_like(tmp)
        ret = tensor.set_subtensor(ret[:, :-1], tmp[:, 1:])
        return tensor.cast(ret, theano.config.floatX)

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
    return raw_pred

