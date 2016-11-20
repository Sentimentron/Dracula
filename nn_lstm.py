"""

    This file defines the LSTM layer of the neural network.

"""

import theano
from theano import tensor
from util import numpy_floatX

def lstm_unmasked_layer(tparams, state_below, options, prefix='lstm', mult=1, go_backwards=False):
    """

    :param tparams:
    :param state_below:
    :param options:
    :param prefix:
    :return:
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _p(pp, name):
        return '%s_%s' % (pp, name)

    def _step(x_, h_, c_):

        U = tparams[_p(prefix, 'U')]
#        h_ = theano.printing.Print("h_", attrs=["shape"])(h_)
#        U = theano.printing.Print("U_", attrs=["shape"])(U)
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['lstm_proj']*mult))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['lstm_proj']*mult))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['lstm_proj']*mult))
        c = tensor.tanh(_slice(preact, 3, options['lstm_proj']*mult))

        c = f * c_ + i * c

        h = o * tensor.tanh(c)

        return h, c

 #   state_below = theano.printing.Print("state_below", attrs=["shape"])(state_below)
    W = tparams[_p(prefix, 'W')]
    b = tparams[_p(prefix, 'b')]

#    W = theano.printing.Print("W", attrs=["shape"])(W)
#    b = theano.printing.Print("W", attrs=["shape"])(b)
    state_below = tensor.dot(state_below, W) + b

    lstm_proj = options['lstm_proj']*mult
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[tensor.cast(tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           lstm_proj), theano.config.floatX),
                                              tensor.cast(tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           lstm_proj), theano.config.floatX)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                go_backwards=go_backwards)
    return rval[0]


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, go_backwards=False, mult=1):
    """

    :param tparams:
    :param state_below:
    :param options:
    :param prefix:
    :param mask:
    :return:
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _p(pp, name):
        return '%s_%s' % (pp, name)

    def _step(m_, x_, h_, c_):

        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['lstm_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['lstm_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['lstm_proj']))
        c = tensor.tanh(_slice(preact, 3, options['lstm_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    lstm_proj = options['lstm_proj'] * mult
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           lstm_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           lstm_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, go_backwards=go_backwards)
    return rval[0]

def bidirectional_lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, mult=1):

    def _p(pp, name):
        return '%s_%s' % (pp, name)

    prefix_forwards = '%s_forwards' % (prefix,)
    prefix_backwards = '%s_backwards' % (prefix,)

    if mask is not None:
        forwards = lstm_layer(tparams, state_below, options, prefix=prefix_forwards, mask=mask, go_backwards=False, mult=mult)
        backwards = lstm_layer(tparams, state_below, options, prefix=prefix_backwards, mask=mask, go_backwards=True, mult=mult)
    else:
        forwards = lstm_unmasked_layer(tparams, state_below, options, prefix=prefix_forwards, mult=mult, go_backwards=False)
        backwards = lstm_unmasked_layer(tparams, state_below, options, prefix=prefix_backwards, mult=mult, go_backwards=True)

    #forwards = theano.printing.Print(prefix_forwards, attrs=["shape"])(forwards)
    #backwards = theano.printing.Print(prefix_forwards, attrs=["shape"])(backwards)

    return forwards + backwards

