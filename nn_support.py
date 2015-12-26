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
        x, mask, wmask, y, y_mask = prepare_data([data[0][t] for t in valid_index],
                                                 numpy.array(data[1])[valid_index],
                                                 maxlen=maxlen)
        prediction_probs = f_pred_prob(x, mask, wmask, y_mask)
        probs[valid_index, :] = prediction_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, maxlen, maxw, n_proj):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = []
    valid_shapes = []
    for _, valid_index in iterator:
        xc, xw, mask, wmask, y, y_mask = prepare_data([data[0][t] for t in valid_index],
                                                 [data[1][t] for t in valid_index],
                                                 numpy.array(data[2])[valid_index],
                                                 maxlen, maxw, n_proj)
        preds = f_pred(xc, mask, wmask, y_mask)
        preds = preds[numpy.nonzero(y)]
        acc = numpy.equal(preds, y[numpy.nonzero(y)])
        valid_shapes.append(preds.size)
        valid_err.append(acc.sum())

    valid_err = 1. - 1.0*numpy.asarray(valid_err).sum() / numpy.asarray(valid_shapes).sum()

    return valid_err
