"""
    This file contains the rest of the layers in the network.
"""

import theano
import numpy as np
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
    # Used for masking only
    zeros_layer = tensor.alloc(numpy_floatX(0.), 16, n_samples, dim)
    avg_layer = tensor.alloc(numpy_floatX(0.), 16, n_samples, dim)
    # HACK: just make this much larger than any value we're likely to encounter
    min_layer = tensor.alloc(numpy_floatX(10000.), 16, n_samples, dim)
    min_layer_comp = tensor.alloc(numpy_floatX(10000.), 16, n_samples, dim) # Used for masking
    max_layer = tensor.alloc(numpy_floatX(-10000.), 16, n_samples, dim)
    max_layer_comp = tensor.alloc(numpy_floatX(-10000.), 16, n_samples, dim) # Used for masking
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

    def min_value_at_position(location, output_model, values):
        output_subtensor = output_model[location[0], location[2]]
        values_subtensor = values[location[3], location[2]]
        return tensor.set_subtensor(output_subtensor,
                                    tensor.switch(
                                        tensor.lt(values_subtensor, output_subtensor),
                                        values_subtensor, output_subtensor
                                        )
                                    )

    def max_value_at_position(location, output_model, values):
        output_subtensor = output_model[location[0], location[2]]
        values_subtensor = values[location[3], location[2]]
        return tensor.set_subtensor(output_subtensor,
                                    tensor.switch(
                                        tensor.gt(values_subtensor, output_subtensor),
                                        values_subtensor, output_subtensor
                                        )
                                    )

    (avg_layer, count_layer), _ = theano.foldl(fn=set_value_at_position,
                                               sequences=[wmask],
                                               outputs_info=[avg_layer, count_layer],
                                               non_sequences=[fixed_ones, proj]
                                               )
    min_layer, _ = theano.foldl(fn=min_value_at_position,
                                sequences=[wmask],
                                outputs_info=[min_layer],
                                non_sequences=[proj])

    max_layer, _ = theano.foldl(fn=max_value_at_position,
                                sequences=[wmask],
                                outputs_info=[max_layer],
                                non_sequences=[proj])

    min_layer = tensor.switch(tensor.eq(min_layer, min_layer_comp), zeros_layer, min_layer)
    max_layer = tensor.switch(tensor.eq(max_layer, max_layer_comp), zeros_layer, max_layer)

    count_layer_inverse = 1.0 / count_layer
    count_layer_mask = 1.0 - tensor.isinf(count_layer_inverse)
    count_layer_mult = tensor.isinf(count_layer_inverse) + count_layer
    avg_layer = (avg_layer / count_layer_mult) * count_layer_mask

    output = tensor.concatenate([avg_layer, min_layer, max_layer], axis=2)
    return output


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
