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

def init_params(options, reloaded=False):
    """
    Global (not LSTM) parameter. For the embedding and the classifier.
    """
    params = OrderedDict()

    # Embedding setup
    options['dim_proj'] = options['dim_proj_chars']# + options['dim_proj_words']
    # Use valid to compute the border, so it generates output of
    # input_shape - filter_shape - 1
    lstm_proj = 5 * (options['dim_proj'] - 5 + 1) * (options['max_letters'] - 5 + 1)
    options['lstm_proj'] = int(lstm_proj)
    logging.debug("dim_proj = %d", options['dim_proj'])

    if not reloaded:
        nparams = generate_init_params(options, params)
        return nparams
    else:
        return options

def generate_init_params(options, params):

    randn = numpy.random.rand(options['n_chars'],
                              options['dim_proj_chars'])*2 - 1
    params['Cemb'] = (1 * randn).astype(config.floatX)

    # 5 x 5 2D convolution, done 5 times
    params['conv'] = 0.01 * numpy.random.randn(5, 1, 5, 5).astype(config.floatX)

    for i in range(options['word_layers']):
        name = 'lstm_words_%d' % (i + 1,)
        params = param_init_bidirection_lstm(options, params, prefix=name, proj=options['lstm_proj'])

    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['lstm_proj'],
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


def param_init_lstm(options, params, prefix='lstm', proj=1):
    """
    Init the LSTM parameter:

    :see: init_params
    """

    W = numpy.concatenate([ortho_weight(proj),
                           ortho_weight(proj),
                           ortho_weight(proj),
                           ortho_weight(proj)], axis=1)
    U = numpy.concatenate([ortho_weight(proj),
                           ortho_weight(proj),
                           ortho_weight(proj),
                           ortho_weight(proj)], axis=1)
    b = numpy.zeros((4 * proj,))
    print(b.shape)

    params[_p(prefix, 'W')] = W.astype(config.floatX)
    params[_p(prefix, 'U')] = U.astype(config.floatX)
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def param_init_bidirection_lstm(options, params, prefix='lstm', proj=1):
    prefix_forwards = '%s_forwards' % (prefix,)
    prefix_backwards = '%s_backwards' % (prefix,)

    params = param_init_lstm(options, params, prefix_forwards, proj)
    params = param_init_lstm(options, params, prefix_backwards, proj)

    return params
