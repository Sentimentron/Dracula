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
import unicodecsv as csv

def nearest_power_of_two(of):
    def find_nearest(array, value):
        a = numpy.abs(array - value)
        idx = a.argmin()
        val = a[idx]
        if val == a[idx+1]:
            # If equidistant, tip the balance in favour of
            # the larger power of two
            return array[idx+1]
        return array[idx]

        idx = (numpy.abs(array - value)).argmin()
        return array[idx]
    pows_of_two = numpy.power(2, range(9))
    return find_nearest(pows_of_two, of)

def build_character_dictionary(path, chars = {}):
    with io.open(path, mode='r') as fin:
        filereader = csv.reader(fin, delimiter='\t')
        for c, (_, _, _, q1, q2, dup) in enumerate(filereader):
            text = q1.split() + q2.split()
            for word in text:
                for c in word:
                    if c not in chars:
                        chars[c] = len(chars) + 1
    return chars

def get_tweet_words(path):
    t = defaultdict(list)
    with io.open(path, mode='r', encoding='utf8') as fin:
        filereader = csv.reader(fin, delimiter='\t')
        for c, (_, _, _, q1, q2, dup) in enumerate(filereader):
            text = q1.split()
            for word in text:
                t[2*c].append(word)
            text = q2.split()
            for word in text:
                t[2*c + 1].append(word)
    return t

def get_max_word_count(path):
    t = get_tweet_words(path)
    m = [len(t[c]) for c in t]
    m = int(numpy.percentile(m, 99))
    #m = int(numpy.median([len(t[c]) for c in t]))
    m = nearest_power_of_two(m)
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
    chars1, chars2, labels = [], [], []
    with open(path, 'r') as fin:
        filereader = csv.reader(fin, delimiter='\t')
        for _, _, _, q1, q2, dup in filereader:
            buf = []
            text = q1.split()
            for i, word in enumerate(text):
                cur_chars = []
                for j, c in enumerate(word):
                    cidx = chardict[c] if c in chardict else 0
                    cur_chars.append(cidx)
                buf.append(cur_chars)
            chars1.append(buf)
            buf = []
            text = q2.split()
            for i, word in enumerate(text):
                cur_chars = []
                for j, c in enumerate(word):
                    cidx = chardict[c] if c in chardict else 0
                    cur_chars.append(cidx)
                buf.append(cur_chars)
            chars2.append(buf)
            if dup == "0":
                labels.append(1)
            elif dup == "1":
                labels.append(0)
    return chars1, chars2, labels


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
    y = numpy.zeros(n_samples).astype('int8')
    y_mask = numpy.zeros(n_samples).astype('float32')

    for idx, (s_c, l) in enumerate(zip(char_seqs, labels)):
        # idx is the current position in the mini-batch
        # s_c is a 2D list of characters
        # l is the current label

        # Set the y-label
        y[idx] = l
        y_mask[idx] = 1

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
