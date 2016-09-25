"""

	Contains I/O functions

"""

import io
import numpy
import theano
import logging
import sys
import tempfile

from collections import defaultdict
import csv


def build_character_dictionary(path, chars = {}):
    with io.open(path, mode='r', encoding='utf8') as fin:
        filereader = csv.reader(fin)
        for text, polarity in filereader:
            text = text.split()
            for word in text:
                for c in word:
                    if c not in chars:
                        chars[c] = len(chars) + 1
    return chars

def get_tweet_words(path):
    t = defaultdict(list)
    with io.open(path, mode='r', encoding='utf8') as fin:
        filereader = csv.reader(fin)
        for c, (text, _) in enumerate(filereader):
            text = text.split()
            for word in text:
                t[c].append(word)
    return t

def get_max_word_count(path):
    t = get_tweet_words(path)
    m = [len(t[c]) for c in t]
    m = int(numpy.percentile(m, 99))
    #m = int(numpy.median([len(t[c]) for c in t]))
    logging.debug("get_max_word_count('%s') = %d", path, m)
    return m

def get_max_word_length(path):
    t = get_tweet_words(path)
    m = 0
    d = []
    for c in t:
        for w in t[c]:
            d.append(len(w))
            if len(w) >= m:
                m = len(w)
                logging.debug('length: %s, %d', w, m)
    m = numpy.percentile(d, 99)
    logging.debug("get_max_word_length('%s') = %d", path, m)
    return m

def get_max_length(path):
    t = get_tweet_words(path)
    t = {c: u"".join(t[c]) for c in t}
    m = max([len(t[c]) for c in t])
    logging.debug('get_max_length(%s) = %d', path, m)
    return m

def load_data(path, chardict = {}, allow_append=True):

    if allow_append:
        build_character_dictionary(path, chardict)

    cur_chars = []
    chars, labels = [], []
    with open(path, 'r') as fin:
        filereader = csv.reader(fin)
        for text, polarity in filereader:
            buf = []
            text = text.split()
            for i, word in enumerate(text):
                cur_chars = []
                for j, c in enumerate(word):
                    cidx = chardict[c] if c in chardict else 0
                    cur_chars.append(cidx)
                buf.append(cur_chars)
            chars.append(buf)
            labels.append(float(polarity))
    return chars, labels


def string_to_unprepared_format(text, chardict):

    chars, labels, buf = [], [], []
    text = text.split()
    for i, word in enumerate(text):
        cur_chars = []
        for j, c in enumerate(word):
            cidx = chardict[c] if c in chardict else 0
            cur_chars.append(cidx)
        buf.append(cur_chars)
    chars.append(buf)
    labels.append(0)
    return chars, labels


def prepare_data(char_seqs, labels, maxw, maxwlen, dim_proj):

    # x: a list of sentences
    n_samples = len(char_seqs)

    x_c = numpy.zeros((maxw, maxwlen, n_samples)).astype('int8')
    x_mask = numpy.zeros((maxw, maxwlen, n_samples, dim_proj)).astype(theano.config.floatX)
    y = numpy.zeros((1, n_samples)).astype('float32')
    y_mask = numpy.zeros((1, n_samples)).astype('float32')

    for idx, (s_c, l) in enumerate(zip(char_seqs, labels)):
        # idx is the current position in the mini-batch
        # s_c is a 2D list of characters
        # l is the current label

        # Set the y-label
        y[0, idx] = l
        y_mask[0, idx] = 1.

        warning = None

        for c, warr in enumerate(s_c):
            # idx is the current tweet in this minibatch
            # c is the current word
            if c >= maxw:
                warning = "truncation: too many words in this tweet!"
                break
            for p, carr in enumerate(warr): # huh! good God!
                # p is the current character in this word
                if p >= maxwlen:
                    warning = "truncation too many chars in this word!"
                    break
                x_c[c, p, idx] = carr
                x_mask[c, p, idx] = numpy.ones(dim_proj)

    return x_c, x_mask, y, y_mask
