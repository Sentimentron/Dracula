"""
    nn_support.py

    Contains evaluation/prediction functions.

"""

import numpy
import theano


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False, maxlen=140):
    """
    If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(theano.config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x0, mask0, wmask, y, y_mask = prepare_data([data[0][t] for t in valid_index],
                                                 numpy.array(data[2])[valid_index],
                                                 maxlen=maxlen)
        x1, mask0, wmask, y, y_mask = prepare_data([data[1][t] for t in valid_index],
                                                 numpy.array(data[2])[valid_index],
                                                 maxlen=maxlen)
        prediction_probs = f_pred_prob(x0, x1, mask0, mask1, wmask, y_mask)
        probs[valid_index, :] = prediction_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, maxw, max_word_length, dim_proj):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = []
    valid_shapes = []
    for _, valid_index in iterator:
        xc0, mask0, y, y_mask = prepare_data([data[0][t] for t in valid_index],
                                              numpy.array(data[2])[valid_index],
                                              maxw, max_word_length, dim_proj)
        xc1, mask1, y, y_mask = prepare_data([data[1][t] for t in valid_index],
                                              numpy.array(data[2])[valid_index],
                                              maxw, max_word_length, dim_proj)
        preds = f_pred(xc0, xc1, mask0, mask1, y_mask)
        acc = numpy.equal(preds, y)
        valid_shapes.append(preds.size)
        valid_err.append(acc.sum())

    valid_err = 1. - 1.0*numpy.asarray(valid_err).sum() / numpy.asarray(valid_shapes).sum()

    return valid_err
