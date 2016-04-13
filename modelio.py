"""

	Contains I/O functions

"""

import numpy
import theano
import logging
import sys
import tempfile

from collections import defaultdict

def build_character_dictionary(path, chars = {}):
    with open(path, 'r') as fin:
        lineno = 0
        for line in fin:
            lineno += 1
            line = line.strip()
            if len(line) == 0:
                continue
            try:
                word, _ = line.split('\t')
                for c in word:
                    if c not in chars:
                        chars[c] = len(chars) + 1
            except ValueError as ex:
                print ex, lineno, line
    return chars

def build_word_dictionary(path, words = {}):
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            word, _ = line.split('\t')
            if word not in words:
                words[word] = len(words) + 1
    return words

def build_tag_dictionary(path, tags={}):
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            _, tag = line.split('\t')
            if tag not in tags:
                tags[tag] = len(tags) + 1
    return tags

def get_tweet_words(path):
    t = defaultdict(list)
    c = 0
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                c += 1
                continue
            word, pos = line.split('\t')
            word = word.decode('utf8')
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

def load_pos_tagged_data(path, chardict = {}, worddict={}, posdict={}, overlap=15, allow_append=True):

    if allow_append:
        build_character_dictionary(path, chardict)
        build_word_dictionary(path, worddict)
        build_tag_dictionary(path, posdict)

    cur_chars, cur_words, cur_labels = [], [], []
    words, chars, labels = [], [], []
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                chars.append(cur_chars[:-1])
                words.append(cur_words[:-1])
                labels.append(cur_labels)
                cur_chars = []
                cur_labels = []
                cur_words = []
                continue

            word, pos = line.split('\t')

            if word not in worddict and allow_append:
                worddict[word] = len(worddict)+1

            for c in word:
                if c not in chardict and allow_append:
                    chardict[c] = len(chardict)+1

                if c in chardict:
                    cur_chars.append(chardict[c])
                else:
                    cur_chars.append(0)

                if word in worddict:
                    cur_words.append(worddict[word])
                else:
                    cur_words.append(0)

                if pos not in posdict and allow_append:
                    posdict[pos] = len(posdict)+1

            if pos in posdict:
                cur_labels.append(posdict[pos])
            else:
                cur_labels.append(0)

            if word in worddict:
                cur_words.append(worddict[word])
            else:
                cur_words.append(0)
            cur_chars.append(0)

    if len(cur_chars) > 0:
    	chars.append(cur_chars)
        words.append(cur_words)
    	labels.append(cur_labels)

    return chars, words, labels

def string_to_unprepared_format(text, chardict, worddict):

    with open('sample.conll', 'wb') as fp:
        for word in text.split():
            #if word not in worddict:
            #    raise Exception((word, "not in dictionary"))
            line = '%s\t?\n' % (word,)
            fp.write(line)
            #           print >> fp, "%s\t?" % (word,)

    chars, words, labels = load_pos_tagged_data("sample.conll", chardict, worddict, {'?': 0}, False)
    return [], chars, words, labels

def prepare_data(char_seqs, labels, maxw, maxwlen, dim_proj):
    """
    Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequences to this maximum
    length

    This swap the axis!
    """

    # x: a list of sentences
    n_samples = len(char_seqs)

    x_c = numpy.zeros((maxw, maxwlen, n_samples)).astype('int8')
    x_mask = numpy.zeros((maxw, maxwlen, n_samples, dim_proj)).astype(theano.config.floatX)
    y = numpy.zeros((maxw, n_samples)).astype('int8')
    y_mask = numpy.zeros((maxw, n_samples)).astype('int8')

    for idx, (s_c, l) in enumerate(zip(char_seqs, labels)):
        # idx is the current position in the mini-batch
        # s_c is a list of characters
        # s_w is a list of words
        # l is a list of labels
        c = 0
        p = 0
        warning = None
        for j, a in enumerate(s_c):
            # j is the current character in this tweet
            # idx is the current tweet in this minibatch
            # c is the current word (can be up to 16)
            # p is the current character in this word

            if a == 0:
                # This current character is a space
                # Increase the word count and continue
                c += 1
                p = 0
                j += 1 # Temporarily skip to next loop char
                if c >= maxw:
                    if j != len(s_c):
                        warning = "truncation: too many words in this tweet! {}-{}".format(j, len(s_c))
                    break
                if c >= len(l):
                    if j != len(s_c):
                        warning = "truncation: too many words for these labels {}-{}".format(j, len(s_c))
                    break

            if p >= x_c.shape[1]:
                warning = "truncation: too many characters for this maxwlen"
            else:
                x_c[c, p, idx] = a
                x_mask[c, p, idx] = numpy.ones(dim_proj)

            y[c, idx] = l[c]
            y_mask[c, idx] = 1
            p += 1

        if warning is not None:
            #logging.warning("%s", warning)
            pass

    return x_c, x_mask, y, y_mask
