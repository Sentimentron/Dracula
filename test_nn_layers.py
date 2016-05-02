import unittest
import theano
import theano.tensor
import numpy
import logging

from nn_layers import per_word_averaging_layer, embeddings_layer

import tensorflow as tf

class WordAveragingOpTests(unittest.TestCase):
    @classmethod
    def get_std(cls):
        maxw, maxwlen = 2, 3
        # n_chars -> maximum character sequence length (e.g. 140)
        # n_samples = minibatch size
        # n_proj = number of combined word/character embeddings
        n_chars, n_samples, n_proj = 4, 2, 4
        # Allocate a matrix which represents the LSTM layer output
        L = numpy.zeros((maxw, maxwlen, n_samples, n_proj))
        L[0, 0, 0, :] = [0.2, 0.3, 0.4, 0.5]
        L[0, 1, 0, :] = [-0.2, -0.4, 0.6, 0.7]
        L[1, 0, 0, :] = [0.8, -0.2, 0.3, -0.1]
        L[0, 0, 1, :] = [-0.4, 0.2, 0.8, 0.9]

        W = numpy.zeros((maxw, maxwlen, n_samples, n_proj))
        W[0, 0, 0, :] = numpy.ones(n_proj) # 1st word, 1st character, 1st tweet
        W[0, 1, 0, :] = numpy.ones(n_proj) # 1st word, 2nd character, 1st tweet
        W[1, 0, 0, :] = numpy.ones(n_proj) # 2nd word, 3rd character, 1st tweet
        W[0, 0, 1, :] = numpy.ones(n_proj) # 1st word, 1st character, 2nd tweet

        # The expected output
        O = numpy.zeros((n_samples, 2, n_proj))
        O[0, 0, :] = [0.0, -0.05, 0.50, 0.60]  # First word, first tweet
        O[1, 0, :] = [0.8, -0.2, 0.3, -0.1]  # Second word, first tweet
        O[0, 1, :] = [-0.4, 0.2, 0.8, 0.9]  # First word, second tweet

        return n_chars, n_samples, n_proj, L, W, O

    def test_embedding_layer(self):
        L = numpy.zeros((4, 4))
        L[0, :] = [0.2, 0.3, 0.4, 0.5]
        L[1, :] = [-0.2, -0.4, 0.6, 0.7]
        L[2, :] = [0.8, -0.2, 0.3, -0.1]
        L[3, :] = [-0.4, 0.2, 0.8, 0.9]

        l = tf.Variable(L, name='emb')

        P = numpy.zeros((4, 4))
        P[0, 0] = 1
        P[0, 1] = 2
        P[0, 3] = 3
        P[1, 0] = 1
        P[1, 1] = 2
        p = tf.constant(P, dtype='int32', name='char')

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        R = embeddings_layer(p, l).eval(session=sess)

        self.assertTrue(numpy.allclose(R[0, 0], [-0.2, -0.4, 0.6, 0.7]))
        self.assertTrue(numpy.allclose(R[0, 1], [0.8, -0.2, 0.3, -0.1]))
        self.assertTrue(numpy.allclose(R[0, 3], [-0.4, 0.2, 0.8, 0.9]))
        self.assertTrue(numpy.allclose(R[1, 0], [-0.2, -0.4, 0.6, 0.7]))
        self.assertTrue(numpy.allclose(R[1, 1], [0.8, -0.2, 0.3, -0.1]))

    def test_forward(self):
        n_chars, n_samples, n_proj, L, W, O = WordAveragingOpTests.get_std()
        LArg = theano.tensor.dtensor4()
        WArg = theano.tensor.dtensor4()

        out = per_word_averaging_layer(LArg, WArg)
        f = theano.function([LArg, WArg], out, on_unused_input='ignore')

        O_actual = f(L, W)

        if False:
            print O.shape, O_actual.shape
            print O
            print O_actual

            for i in [0, 1]:
                for j in  [0, 1]:
                    oact = O_actual[i, j, :]
                    oref = O[i, j, :]
                    if numpy.allclose(oref, numpy.zeros(4)):
                        if numpy.allclose(oact, numpy.zeros(4)):
                            continue
                    print "O =", oact, "ORef =", oref, "i=",i, "j=",j

        self.assertTrue(numpy.allclose(O, O_actual))

if __name__ == "__main__":
    unittest.main()
