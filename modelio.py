"""

	Contains I/O functions

"""

import io
import numpy
import theano
import logging
import sys
import tempfile

from collections import defaultdict, Counter
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
    m = int(numpy.percentile(m, 90))
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
    m = numpy.percentile(d, 90)
    logging.debug("get_max_word_length('%s') = %d", path, m)
    return m

def get_max_length(path):
    t = get_tweet_words(path)
    t = {c: u"".join(t[c]) for c in t}
    m = max([len(t[c]) for c in t])
    logging.debug('get_max_length(%s) = %d', path, m)
    return m

def build_freq_dict(path):
    c = Counter()
    with open(path, 'r') as fin:
        filereader = csv.reader(fin)
        for text, polarity in filereader:
            text = text.split()
            for i, word in enumerate(text):
                c.update([word])

    ret = defaultdict(float)
    total = 0.0
    for i, f in c.most_common():
        total += f
    for i, f in c.most_common():
        ret[i] = int(-numpy.log2(f/total))
    return ret

def load_data(path, chardict = {}, training=False):

    if training:
        freqdict = build_freq_dict(path)
        build_character_dictionary(path, chardict)
    else:
        freqdict = defaultdict(int)

    cur_chars, cur_freqs = [], []
    chars, labels, freqs = [], [], []
    with open(path, 'r') as fin:
        filereader = csv.reader(fin)
        for text, polarity in filereader:
            buf = []
            text = text.split()
            cur_freqs = []
            for i, word in enumerate(text):
                cur_chars = []
                freq = freqdict[word] if word in freqdict else 0
                cur_freqs.append(freq)
                for j, c in enumerate(word):
                    cidx = chardict[c] if c in chardict else 0
                    if training and cidx == 0:
                        raise ValueError("Not permitted")
                    cur_chars.append(cidx)
                buf.append(cur_chars)
            chars.append(buf)
            freqs.append(cur_freqs)
            labels.append(int(float(polarity)))

    if training:
        return chars, labels, freqs

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


def prepare_data(char_seqs, freqs, labels, maxw, maxwlen, dim_proj):

    # x: a list of sentences
    n_samples = len(char_seqs)

    x_c = numpy.zeros((maxw, maxwlen, n_samples)).astype('int8')
    # Note: uses 1s to make sure that we don't get a NaN cost
    f = numpy.ones((maxw, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxw, maxwlen, n_samples, dim_proj)).astype(theano.config.floatX)
    y = numpy.zeros((1, n_samples)).astype('int8')
    y_mask = numpy.zeros((1, n_samples)).astype('int8')

    for idx, (s_c, f_w, l) in enumerate(zip(char_seqs, freqs, labels)):
        # idx is the current position in the mini-batch
        # s_c is a 2D list of characters
        # f_q is a vector of word popularities
        # l is the current label

        # Set the y-label
        y[0, idx] = l + 1 # 2 = positive, 1 = neutral, 0 = negative
        y_mask[0, idx] = 1

        warning = None

        for c, (warr, cf) in enumerate(zip(s_c, f_w)):
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
            f[c, idx] = cf

    return x_c, f, x_mask, y, y_mask
