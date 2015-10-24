"""

    Contains code for initializing the LSTM parameters.

"""

from collections import OrderedDict
import logging

import numpy
import theano
from theano import config


def _p(pp, name):
    return '%s_%s' % (pp, name)

def init_params(options):
    """
    Global (not LSTM) parameter. For the embedding and the classifier.
    """
    params = OrderedDict()

    # Embedding setup
    logging.debug("dim_proj_chars = %d, dim_proj_words = %d", options['dim_proj_chars'], options['dim_proj_words'])
    options['dim_proj'] = options['dim_proj_chars']# + options['dim_proj_words']
    logging.debug("dim_proj = %d", options['dim_proj'])

    randn = numpy.random.rand(options['n_chars'],
                              options['dim_proj_chars'])*2 - 1
    params['Cemb'] = (0.01 * randn).astype(config.floatX)

    #randn = numpy.random.rand(options['n_words'],
    #                          options['dim_proj_words'])
    #params['Wemb'] = (0.01 * randn).astype(config.floatX)*2 - 1


    params = param_init_lstm(options,
                             params,
                             prefix="lstm")

    params = param_init_lstm(options,
                             params,
                             prefix="lstm_words")

    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm', mult=1):
    """
    Init the LSTM parameter:

    :see: init_params
    """

    W = numpy.concatenate([ortho_weight(options['dim_proj']*mult),
                           ortho_weight(options['dim_proj']*mult),
                           ortho_weight(options['dim_proj']*mult),
                           ortho_weight(options['dim_proj']*mult)], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']*mult),
                           ortho_weight(options['dim_proj']*mult),
                           ortho_weight(options['dim_proj']*mult),
                           ortho_weight(options['dim_proj']*mult)], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'] * mult,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params
