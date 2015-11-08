"""

    Contains functions for saving and retrieving the model.

"""

from collections import OrderedDict
import logging

import numpy


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def load_params(path, params):
    logging.info("Loading model from file '%s'...", path)
    pp = numpy.load(path)
    for k in pp:
        if k in ['Cemb', 'Wemb', 'lstm_W', 'lstm_b', 'lstm_U', 'lstm_words_W', 'lstm_words_b', 'lstm_words_U', 'U', 'b']:
            params[k] = pp[k]
    return params
