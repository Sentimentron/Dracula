"""

    Contains functions for saving and retrieving the model.

"""

from collections import OrderedDict
import logging
import numpy
import pickle


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
        params[k] = pp[k]
    path = "%s.pkl" % (path,)
    logging.info("Loading model from file '%s'...", path)
    with open(path, 'rb') as fin:
        data = pickle.load(fin)
        for k in ['dim_proj_chars', 'char_dict']:
            params[k] = data[k]
    return params
