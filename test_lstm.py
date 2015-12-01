__author__ = 'rtownsend'

import tensorflow as tf
import unittest
import numpy

from nn_lstm import LSTMOutputLayer

class LSTMTests(unittest.TestCase):

    def test_very_basic_lstm_layer(self):
        """
        1) Set up an embeddings matrix containing two tweet reps:
            "hello" and "lollipop"
        2) Put an LSTM layer on top
        3) Put a soft-max layer on top of that and try to predict
           where the "l"s are.
        This is so basic that it doesn't need any kind of recurrence,
        it's just checking we have the syntax right.
        :return:
        """

        embeddings = dict()
        embeddings['l'] = numpy.asarray([-0.4, 0.5])
        for i in ['h', 'e', 'o', 'i', 'p']:
            while True:
                embeddings[i] = numpy.random.uniform(-1, 1, 2)
                if numpy.sum(numpy.power(embeddings[i] - embeddings['l'], 2)) > 1.0:
                    break

        # n_chars, batch_size, n_proj
        embeddings_layer = numpy.zeros((140, 2, 2), dtype='float32')
        targets = numpy.zeros((140, 2), dtype='float32')
        tweets = ["hello", "lollipop"]
        for j, t in enumerate(tweets):
            for i, l in enumerate(t):
                assert l in embeddings
                embeddings_layer[i, j, :] = embeddings[l]
                if l == 'l':
                    targets[i, j] = 1.

        # OK: now create the embeddings layer
        batch_size = 2
        lstm_state_size = 140

        cell = rnn_cell.BasicLSTMCell(lstm_state_size)
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [lstm_state_size, batch_size])
            softmax_b = tf.get_variable("softmax_b", [batch_size])

        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))

        inputs = tf.split(0, 140, embeddings_layer_tf)
        inputs = [tf.squeeze(i, [0]) for i in inputs]

        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, states = seq2seq.rnn_decoder(inputs, initial_state, loop_function=loop, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, lstm_state_size])


        embeddings_layer_tf = tf.constant(embeddings_layer, dtype='float32')
        encoder_inputs = []
        for i in range(140):
            encoder_inputs.append(tf.placeholder(tf.float32, shape=[2, 2], name="encoder{0}".format(i)))

        decoder_inputs = []
        for i in range(140):
            decoder_inputs.append(tf.placeholder(tf.float32, shape=[2, 2], name="decoder{0}".format(i)))

        targets = [decoder_inputs[i + 1] for i in xrange(len(decoder_inputs) - 1)]

        outputs, losses = seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, rnn_cell.BasicLSTMCell(140))

        softmax_w = tf.random_uniform((140, 2))
        softmax_b = tf.random_uniform(140)

        output, state = seq2seq.basic_rnn_seq2seq()

        output, state = rnn.rnn(rnn_cell.BasicLSTMCell(lstm_state_size), [tf.squeeze(i, [0]) for i in tf.split(0, 140, embeddings_layer_tf)], dtype='float32')
        #output = [tf.expand_dims(i, 2) for i in output]
        #output = tf.concat(2, output)
        print output
        logits = tf.matmul(output[-1], softmax_w) + softmax_b
        probabilities = tf.nn.softmax(logits)
        preds = tf.argmax(probabilities, 0)
        loss = tf.reduce_mean(tf.square(preds - targets))

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        print loss.eval(session=sess)