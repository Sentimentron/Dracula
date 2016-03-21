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
    logging.debug("dim_proj = %d", options['dim_proj'])

    nparams = generate_init_params(options, params)

    if not reloaded:
        return nparams
    else:
        for k in nparams:
            if k in options:
                nparams[k] = options[k]
        logging.debug("%s %s", options.keys(), nparams.keys())
        return nparams

def generate_init_params(options, params):

    randn = numpy.random.rand(options['n_chars'],
                              options['dim_proj_chars'])*2 - 1
    params['Cemb'] = (0.01 * randn).astype(config.floatX)

    params = param_init_bidirection_lstm(options,
                                         params,
                                         prefix="lstm_chars_1")

    params = param_init_bidirection_lstm(options,
                                         params,
                                         prefix="lstm_chars_2")

    params = param_init_bidirection_lstm(options,
                                         params,
                                         prefix="lstm_chars_3")

    params = param_init_bidirection_lstm(options,
                                         params,
                                         prefix="lstm_words_1", mult=1)

    params = param_init_bidirection_lstm(options,
                                         params,
                                         prefix="lstm_words_2", mult=1)

    params = param_init_bidirection_lstm(options,
                                         params,
                                         prefix="lstm_words_3", mult=1)

    params = param_init_bidirection_lstm(options,
                                         params,
                                         prefix="lstm_words_4", mult=1)

    params = param_init_bidirection_lstm(options,
                                         params,
                                         prefix="lstm_words_5", mult=1)

    params = param_init_bidirection_lstm(options,
                                         params,
                                         prefix="lstm_words_6", mult=1)

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
    params[_p(prefix, 'W')] = W.astype(config.floatX)
    U = numpy.concatenate([ortho_weight(options['dim_proj']*mult),
                           ortho_weight(options['dim_proj']*mult),
                           ortho_weight(options['dim_proj']*mult),
                           ortho_weight(options['dim_proj']*mult)], axis=1)
    params[_p(prefix, 'U')] = U.astype(config.floatX)
    b = numpy.zeros((4 * options['dim_proj'] * mult,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def param_init_bidirection_lstm(options, params, prefix='lstm', mult=1):
    prefix_forwards = '%s_forwards' % (prefix,)
    prefix_backwards = '%s_backwards' % (prefix,)

    params = param_init_lstm(options, params, prefix_forwards, mult)
    params = param_init_lstm(options, params, prefix_backwards, mult)

    return params
