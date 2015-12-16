import numpy as np
import tensorflow as tf

import cPickle as pickle
import argparse

from modelio import *
from nn_layers import *
from nn_lstm import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="Data/TweeboOct27.conll",
                        help="Path to input (CONLL-format)")

    parser.add_argument("--batch_size", type=int, default=20, help="mini-batch size")

    parser.add_argument("--dim", type=int, default=12, help="Size of the embedding matrix")

    args = parser.parse_args()
    train(args)

class Dracula(object):

    def __init__(self, args, wemb, cemb, n_words, n_chars, n_max_offset, n_pos):

        # Embeddings layer
        self.wemb = tf.Variable(wemb, trainable=True)
        self.cemb = tf.Variable(cemb, trainable=True)

        self.wmask = tf.placeholder(tf.float32, [n_words, args.batch_size, n_chars, args.dim])

        self.input_chars = tf.placeholder(tf.int32, [args.batch_size, n_chars])
        self.input_words = tf.placeholder(tf.int32, [args.batch_size, n_chars])

        self.input_lstm = tf.concat(1,
                                    embeddings_layer(self.input_chars, self.cemb),
                                    embeddings_layer(self.input_words, self.wemb))

        self.first_lstm = LSTMLayer(model='lstm', n_chars, args.batch_size, n_chars, args.dim, name='lstm_first')

        self.averaging_layer = per_word_averaging_layer(self.first_lstm.outputs, self.wmask, n_max_offset)

        self.output_lstm = LSTMOutputLayer(model='lstm', n_chars, args.batch_size, n_chars, args.dim, name='lstm_final',
                                           n_pos, infer=False)


def train(args):
    # Load the data
    n_chars = get_max_words_in_tweet(args.input)
    chardict = build_character_dictionary(args.input)
    worddict = build_word_dictionary(args.input)
    tagdict = build_tag_dictionary(args.input)
    maxoffset = get_max_word_offset(args.input)

    with open('dicts.pkl', 'w') as f:
        pickle.dump((chardict, worddict, tagdict), f)

    wmask, chars, words, tags = read_data(args.input, args.dim, chardict, worddict, tagdict)

    # Build the model
    wemb = tf.random_uniform((len(worddict), args.dim), name="wemb")
    cemb = tf.random_uniform((len(chardict), args.dim), name="cemb")

    model = Dracula(args, wemb, cemb, wmask.shape[0], wmask.shape[1], maxoffset, len(tagdict)+1)

    with tf.Session() as sess:
        