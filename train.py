#!/bin/env python

"""
Main training script
"""

import os
from collections import OrderedDict
from tag import IDENT
import pickle

import theano
import theano.tensor as T
import numpy as np

import logging

class HiddenLayer(object):

    def __init__(self, input, dim1, dim2):
        """Create the hidden layer"""
        self.W = theano.shared(name='W',
                               value=np.random.uniform(-0.2, 0.2, (dim1 * dim2))
                               .astype(theano.config.floatX))

        self.b = theano.shared(name='b',
                               value=np.zeros(dim2,).astype(theano.config.floatX))


        self.output = T.tanh(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, dim0, dim1, dim2):
        self.emb = theano.shared(name='embeddings',
                                 value=np.random.uniform(-0.2, 0.2, (dim0, dim1))
                                 .astype(theano.config.floatX))
        self.x = T.vector('x', dtype='int32')
        self.e = self.emb[self.x].reshape([dim0 * dim1])
        self.hidden = HiddenLayer(self.e, dim0*dim1, dim2)
        self.output = T.nnet.softmax(self.hidden.output)

        self.forward = theano.function(inputs=[self.x], outputs=[self.output])





class RNN(object):
    """
    The Recurrent Neural Network model (Elman)

        See http://deeplearning.net/tutorial/rnnslu.html
    """

    def __init__(self, nh, nc, ne, de, cs):
        """
        Construct and configure the network
        :param nh: size of the hidden layer
        :param nc: number of output classes
        :param ne: number of characters to consider
        :param de: word embedding dimensions
        :param cs: maximum context to consider when predicting
        :return:
        """

        # Character-space embeddings
        self.emb = theano.shared(name='embeddings',
                                 value=np.random.uniform(-0.2, 0.2,
                                                           (ne+1, de)
                                                           ).astype(theano.config.floatX),
                                 )
        # First layer weights
        self.wx = theano.shared(name='wx',
                                value=np.random.uniform(-0.2, 0.2,
                                                        (de * cs, nh)))

        # Hidden layer
        self.wh = theano.shared(name='wh',
                                value=np.random.uniform(-0.2, 0.2,
                                                        (nh, nh)))

        # Activation layer
        self.w = theano.shared(name='w',
                               value=np.random.uniform(-0.2, 0.2,
                                                       (nh, nc)))

        # Hidden layer bias
        self.bh = theano.shared(name='bh',
                                value=np.zeros(nh,
                                                  dtype=theano.config.floatX))

        # Activation layer bias
        self.b = theano.shared(name='b',
                               value=np.zeros(nc,
                                              dtype=theano.config.floatX))

        # Previous hidden layer
        self.h0 = theano.shared(name='h0',
                                value=np.zeros(nh,
                                                  dtype=theano.config.floatX))

        self.params = [self.emb, self.wx, self.wh, self.w, self.bh, self.b, self.h0]
        self.names = ['emb', 'wx', 'wh', 'w', 'bh', 'b', 'h0']

        #idxs = T.imatrix()
        #x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        x = T.matrix('x', dtype='int64')

        y = T.vector('y', dtype='int32')

        def recurrence(x_t, h_tm1):
            # Get the input to the hidden layer and the previous layer state
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            # Compute the output classes
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        #[h, s], _ = theano.scan(fn=recurrence,
        #                     sequences=x, outputs_info=[self.h0, None],
        #                     n_steps=x.shape[0])

        def forward(xi):
            embs = self.emb[x.flatten()].reshape([x.shape[0], de])
            h = T.nnet.sigmoid(T.dot(embs, self.wx))
            s = T.nnet.softmax(T.dot(h, self.w))
            return [h, s]

        [h, s] = theano.scan(fn=forward, sequences=x, output_info=[self.h0, None], n_steps=x.shape[0])

        h = T.nnet.sigmoid(T.dot(embs, self.wx))
        a = T.dot(h, self.w)
        logging.debug(a)
        s = T.nnet.softmax(T.dot(h, self.w))

        logging.debug(s)
        y_prob_given_last = s[-1, 0, :]
        y_prob = s[:, 0, :]
        y_pred = T.argmax(y_prob, axis=1)

        # Cost, gradients, learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(y_prob_given_last)[y])
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))

        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.train = theano.function(inputs=[idxs, y, lr], outputs=nll, updates=updates)

        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                      self.emb/T.sqrt(
                                                          (self.emb**2).sum(axis=1)
                                                      ).dimshuffle(0, 'x')
                                                  }
                                         )

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())

def evaluate_accuracy(Y, Ypred):
    """Determines how accurately predicted yPred is"""
    return np.avg(Y == Ypred)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.DEBUG)
    # Load the OCT27 test dataset
    with open('Data/TweeboOct27.pkl', 'r') as fp:
        data = pickle.load(fp)

    # Find the maximum word-length
    wl = max([len(t.word) for t in data])

    # Specify embedding depth
    d = 26

    # Specify context size
    csize = 6

    # Build the number of distinct letters in the set
    dictionary = {}
    for t in data:
        for c in t.word:
            if c not in dictionary:
                dictionary[c] = len(dictionary)+1

    # +1 for the "unknown character" (0)
    num_chars = len(dictionary) + 1

    # Create the thing
    net = RNN(
        nh = 100, # hidden units
        nc = len(IDENT), # class count
        ne = num_chars,
        de = d,
        cs = csize
    )

    # Allocate the input X and Y matrices
    X = np.zeros((csize+1, len(data)), dtype=np.int32)

    # Vectorize
    for i, t in enumerate(data):
        for j, c in enumerate(t.word):
            if j >csize:
                break
            X[j, i] = dictionary[c]
        X[-1, i] = t.tag

    # Shuffle rows
    np.random.shuffle(X)

    # Split into training and validation sets
    # Validation set is ~5% of the data
    val_size = int(0.05*len(data))
    val_set = X[:, :val_size]
    train_set = X[:, val_size+1:]

    max_epochs = 20
    mini_batch_size = 20
    best_accuracy = -np.Inf
    for epoch in range(max_epochs):
        np.random.shuffle(train_set)
        # Divide into mini-batches
        indices = np.arange(0, train_set.shape[1], mini_batch_size)
        for s, e in zip(indices, indices[1:]):
            X = train_set[:-1, s:e]
            y = train_set[-1, s:e].flatten()
            # Run a training pass through the network
            logging.debug(X.shape)
            logging.debug(y.shape)
            net.train(X, y, 0.001)
            net.normalize()

        # Evaluate
        X = val_set[:-1, :]
        Y = val_set[-1, :]
        Ypred = net.classify(X)
        acc = evaluate_accuracy(Y, Ypred)
        logging.info("Epoch %d, accuracy = %.2f", epoch, acc)

        if acc > best_accuracy:
            net.save("Output/")


