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


def per_word_averaging_layer(proj, wmask, n_samples, dim):
    """
    Takes a word-mask and a masked LSTM output, produces a per-word summary.
    :param proj: LSTM output
    :param wmask: word indices matrix
    :param n_samples: maximum number of samples
    :param dim: size of word-embeddings
    :return: per-word average of all character embeddings.
    """
    avg_layer = tensor.alloc(numpy_floatX(0.), 16, n_samples, dim)
    count_layer = tensor.alloc(0, 16, n_samples, dim)
    fixed_ones = tensor.ones_like(count_layer)

    def set_value_at_position(location, output_model, count_model, fixed_ones, values):
        print location.type, values.type, output_model.type
        output_subtensor = output_model[location[0], location[2]]
        count_subtensor = count_model[location[0], location[2]]
        ones_subtensor = fixed_ones[location[0], location[2]]
        values_subtensor = values[location[3], location[2]]
        return tensor.inc_subtensor(output_subtensor, values_subtensor), \
               tensor.inc_subtensor(count_subtensor, ones_subtensor)

    (avg_layer, count_layer), _ = theano.foldl(fn=set_value_at_position,
                                               sequences=[wmask],
                                               outputs_info=[avg_layer, count_layer],
                                               non_sequences=[fixed_ones, proj]
                                               )

    count_layer_inverse = 1.0 / count_layer
    count_layer_mask = 1.0 - tensor.isinf(count_layer_inverse)
    count_layer_mult = tensor.isinf(count_layer_inverse) + count_layer
    return (avg_layer / count_layer_mult) * count_layer_mask


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
    raw_pred, _ = theano.scan(fn=lambda p, free_variable: tensor.nnet.softmax(tensor.dot(p, U * dropout_mask) + b),
                              outputs_info=None,
                              sequences=[avg_per_word, tensor.arange(16)]
                              )

    pred = tensor.zeros_like(raw_pred)
    pred = tensor.inc_subtensor(pred[:, :, 0], 1)
    pred = tensor.set_subtensor(pred[y_mask.nonzero()], raw_pred[y_mask.nonzero()])

    return pred
