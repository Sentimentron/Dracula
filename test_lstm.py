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
        embeddings_layer = numpy.zeros((10, 140, 2), dtype='float32')
        mask = numpy.zeros((10, 140), dtype='float32')
        targets = numpy.zeros((10, 140), dtype='float32')
        tweets = ["hello", "lollipop", "l", "lll", "lop", "poll", "pol", "ill", "lilp", "pill"]
        for j, t in enumerate(tweets):
            for i, l in enumerate(t):
                assert l in embeddings
                embeddings_layer[j, i, :] = embeddings[l]
                mask[j, i] = 1.
                if l == 'l':
                    targets[j, i] = 1.
                    mask[j, i] = 100

        #pred_embeddings_layer = numpy.zeros((140, 2, 2), dtype='float32')
        #tweets = ["olleh", "popillol"]
        #for j, t in enumerate(tweets):
        #    for i, l in enumerate(t):
        #        assert l in embeddings
        #        pred_embeddings_layer[i, j, :] = embeddings[l]

        train_loss = None
        with tf.variable_scope("lstm") as scope:
            layer = LSTMOutputLayer(model='lstm', rnn_size=32, batch_size=10, seq_length=140, n_proj=2, name='lstm_basic',
                       output_size=2, infer=False)
            with tf.Session() as sess:
                tf.initialize_all_variables().run()
                for e in xrange(20):
                    sess.run(tf.assign(layer.lr, tf.constant(0.01 * 15.0/(e + 1))))
                    state = layer.initial_state.eval()
                    x, y = embeddings_layer, targets
                    x = tf.transpose(tf.constant(x, dtype='int32'), [1, 0, 2])
                    y = tf.constant(y, dtype='int32')
                    feed = {layer.input_data: embeddings_layer,
                            layer.targets: targets, layer.initial_state: state,
                            layer.mask: mask}
                    train_loss, state, _ = sess.run([layer.cost, layer.final_state, layer.train_op], feed)
                    print("%.2f" % (train_loss,))
                    pred_layer = layer
                    feed = {pred_layer.input_data: embeddings_layer, pred_layer.initial_state: state}
                    [state] = sess.run([pred_layer.final_state], feed)
                    feed = {pred_layer.input_data: embeddings_layer, pred_layer.initial_state: state}
                    [probs, state] = sess.run([pred_layer.probs, pred_layer.final_state], feed)
                    probs = numpy.reshape(probs, (10, 140, 2))
                    #probs = numpy.transpose(probs, axes=[1, 0, 2])
                    print probs[1], mask[1], probs.shape



        self.assertLess(train_loss, 0.05)

