"""
    nn_support.py

    Contains evaluation/prediction functions.

"""

import numpy
import theano

def pred_error(f_pred, prepare_data, data, iterator, maxw, max_word_length, dim_proj):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = []
    valid_shapes = []
    for _, valid_index in iterator:
        xc, f, mask, y, y_mask = prepare_data([data[0][t] for t in valid_index],
                                                 numpy.array(data[2])[valid_index],
                                                 numpy.array(data[1])[valid_index],
                                                 maxw, max_word_length, dim_proj)
        preds = f_pred(xc, mask, y_mask)
        preds = preds[numpy.nonzero(y_mask)]
        acc = numpy.equal(preds, y[numpy.nonzero(y_mask)])
        valid_shapes.append(preds.size)
        valid_err.append(acc.sum())

    valid_err = 1. - 1.0*numpy.asarray(valid_err).sum() / numpy.asarray(valid_shapes).sum()

    return valid_err
