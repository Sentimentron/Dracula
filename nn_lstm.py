"""

    This file defines the LSTM layer of the neural network.

"""

import theano
from theano import tensor
from util import numpy_floatX
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

def lstm_bidirectional_word_layer(tparams, state_below, options, prefix):

    def _p(pp, name):
        return '%s_%s' % (pp, name)

    lstm_fw_cell = rnn_cell.BasicLSTMCell(32, forget_bias=1.0)
    lstm_bw_cell = rnn_cell.BasicLSTMCell(32, forget_bias=1.0) # TODO: don't hard-code me

    lstm_fw_multicell = rnn_cell.MultiRNNCell([lstm_fw_cell]*options['word_layers'])
    lstm_bw_multicell = rnn_cell.MultiRNNCell([lstm_bw_cell]*options['word_layers'])

    # state_below shape = (max_word_idx, batch_size, dim_proj)

    _X = tf.tranpose(state_below, [1, 0, 2]) # (batch_size, max_word_idx, dim_proj)
    _X = tf.reshape(_X, [-1, 32]) # TODO: don't hard-code me
    _X = tf.split(0, options['max_word_idx'], _X)
    outputs = rnn.bidirectional_rnn(lstm_fw_multicell, lstm_bw_multicell, _X, dtype='float32')[0]

    act = tf.matmul(outputs, tparams[_p(prefix, 'W')]) + tparams[_p(prefix,'b')]
    return act


def lstm_bidirectional_layer(tparams, state_below, options, prefix='lstm'):
    def _p(pp, name):
        return '%s_%s' % (pp, name)

    state_below = tf.Print(state_below, [tf.shape(state_below)], message="state_below=")

    lstm_fw_cell = rnn_cell.BasicLSTMCell(32, forget_bias=1.0)
    lstm_bw_cell = rnn_cell.BasicLSTMCell(32, forget_bias=1.0) # TODO: don't hard-code me

    lstm_fw_multicell = rnn_cell.MultiRNNCell([lstm_fw_cell]*options['letter_layers'])
    lstm_bw_multicell = rnn_cell.MultiRNNCell([lstm_bw_cell]*options['letter_layers'])

    # state_below shape = (max_word_idx, max_char_idx, batch_size, dim_proj)

    def per_word(_X):
        # _X shape = (max_char_idx, batch_size, dim_proj)
        _X = tf.transpose(_X, [1, 0, 2]) # (batch_size, max_char_idx, dim_proj)
        _X = tf.reshape(_X, [-1, 32]) # TODO: don't hard-code me
        _X = tf.split(0, options['max_char_idx'], _X) # (n_steps * (batch_size * n_hidden))
        outputs = rnn.bidirectional_rnn(lstm_fw_multicell, lstm_bw_multicell, _X, dtype='float32')[0]
        print outputs

        def per_word_mul(x):
            print "x", x, type(x)
            per_word_act = tf.matmul(x, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
            per_word_act = tf.Print(per_word_act, [tf.shape(per_word_act)], message="per_word_act=")
            return per_word_act

        per_word_acts = [per_word_mul(x) for x in outputs]
        per_word_acts = [tf.expand_dims(x, 0) for x in per_word_acts]
        per_word_acts = tf.concat(0, per_word_acts)
        per_word_acts = tf.Print(per_word_acts, [tf.shape(per_word_acts)], message="per_word_acts=")
        return per_word_acts


    ret = tf.map_fn(per_word, state_below)
    ret = tf.Print(ret, [tf.shape(ret)], message="ret=")
    return ret
    _X = tf.transpose(state_below, [0, 2, 0]) # (max_word_
    _X = tf.split(0, 50, _X) # TODO: don't hard-code me
    outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X)

    return tf.matmul(outputs[-1], tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]


def lstm_unmasked_layer(tparams, state_below, options, prefix='lstm', mult=1, go_backwards=False):
    """

    :param tparams:
    :param state_below:
    :param options:
    :param prefix:
    :return:
    """
    def _p(pp, name):
        return '%s_%s' % (pp, name)

    state_below = tf.Print(state_below, [tf.shape(state_below)], message="state_below=")

    lstm_cell = rnn_cell.BasicLSTMCell(options['dim_proj'], forget_bias=1.0)

    #state_below = tf.matmul(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    # Convert _X into a list of tensors
    #feed = tf.map_fn(lambda x : x, state_below)
    #feed = tf.unpack(state_below)
    _X = tf.transpose(state_below, [1, 0, 2])
    def preact(_X):
        return tf.matmul(_X, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    _X = tf.map_fn(preact, _X)
    #_X = tf.matmul(_X, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    _X = tf.split(0, 50, _X) # TODO: don't hard-code me

    if go_backwards:
        outputs, states = rnn.rnn(lstm_cell, reverse(_X))
    else:
        outputs, states = rnn.rnn(lstm_cell, list(_X))

    return outputs[-1]

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

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj'] * mult
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
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

