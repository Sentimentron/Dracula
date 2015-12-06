"""

    This file defines the LSTM layer of the neural network.

"""

import theano
from theano import tensor
from util import numpy_floatX

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq

class LSTMLayer(object):

    def __init__(self, model='lstm', rnn_size=140, batch_size=50, seq_length=140, n_proj=16, name='lstm_layer'):

        if model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise ValueError(("model", model))
            return

        self.cell = cell_fn(rnn_size)

        self.input_data = tf.placeholder(tf.float32, [batch_size, seq_length, n_proj], name="input_data")
        self.targets = tf.placeholder(tf.float32, [batch_size, seq_length, n_proj], name="targets")
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        self.inputs = tf.split(1, seq_length, self.input_data)
        self.inputs = [tf.squeeze(input_, [1]) for input_ in self.inputs]

        self.outputs, self.states = seq2seq.rnn_decoder(self.inputs,
                                                        self.initial_state,
                                                        self.cell,
                                                        loop_function=None,
                                                        scope=name)

        self.outputs = tf.concat(0, self.outputs)
        self.final_state = self.states[-1]

    def forward(self, sess, input):
        state = self.cell.zero_state(1, tf.float32).eval()
        feed = {self.input_data: input, self.initial_state: state}
        [state] = sess.run([self.final_state], feed)
        return state

class LSTMOutputLayer(LSTMLayer):

    def __init__(self, model='lstm', rnn_size=140, batch_size=50, seq_length=140, n_proj=16, name='lstm_layer',
                 output_size=2, optimizer=None, infer=False, grad_clip=0.5):
        """
            Initialises the LSTM layer with a softmax class decider
            :param model: Only 'lstm' is supported
            :param rnn_size: The number of timesteps to consider at once
            :param batch_size: Obvious (hopefully) what this is
            :param seq_length: maximum timestep allowed
            :param n_proj: number of embedding dimensions at this stage
            :param name: e.g. 'lstm_layer'
            :param output_size: number of output classes
            :param optimizer: The optimizer to use (default tf.train.AdamOptimizer(self.lr))
            :param infer: whether we're predicting something at the moment or training
            :param grad_clip: Clip gradients to this value (default 0.5) during training.
        """

        super(LSTMOutputLayer, self).__init__(model, rnn_size, batch_size, seq_length, n_proj, name)

        softmax_w = tf.get_variable("{0}_softmax_w".format(name), [rnn_size, output_size])
        softmax_b = tf.get_variable("{0}_softmax_b".format(name), [output_size])

        if infer:
            self.initial_state = self.cell.zero_state(1, tf.float32)
            tf.get_variable_scope().reuse_variables()
            def loop(prev, _):
                prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return prev_symbol
            print self.inputs, self.initial_state, self.cell
            self.outputs, self.states = seq2seq.rnn_decoder(self.inputs, self.initial_state, self.cell, loop_function=loop, scope=name)
            self.outputs = tf.reshape(tf.concat(1, self.outputs))

        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.mask = tf.placeholder(tf.float32, [batch_size, seq_length],name="mask")

        self.logits = tf.nn.xw_plus_b(self.outputs, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [tf.reshape(self.targets, [-1])],
                                                [tf.reshape(self.mask, [-1])],
                                                #[tf.ones([batch_size * seq_length])],
                                                output_size)
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = self.states[-1]
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def infer(self):
        tf.get_variable_scope().reuse_variables()
        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return prev_symbol
        print self.inputs, self.initial_state, self.cell
        outputs, states = seq2seq.rnn_decoder(self.inputs, self.initial_state, self.cell, loop_function=loop, scope="new")


def lstm_unmasked_layer(tparams, state_below, options, prefix='lstm'):
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

        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c

        h = o * tensor.tanh(c)

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[tensor.cast(tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj), theano.config.floatX),
                                              tensor.cast(tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj), theano.config.floatX)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
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

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]
