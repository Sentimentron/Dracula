"""

	Contains I/O functions

"""

import numpy
import theano
import logging
import sys

from collections import defaultdict

def build_character_dictionary(path, chars = {}):
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            word, _ = line.split('\t')
            for c in word:
                if c not in chars:
                    chars[c] = len(chars) + 1
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
            t[c].append(line)
    return t

def get_max_word_count(path):
    t = get_tweet_words(path)
    m = max([len(t[c]) for c in t])
    logging.debug("get_max_word_count('%s') = %d", path, m)
    return m

def get_max_length(path):
    t = get_tweet_words(path)
    t = {c: " ".join(t[c]) for c in t}
    return max([len(t[c]) for c in t])

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
            if word not in worddict:
                raise Exception((word, "not in dictionary"))
            word = unicode(word).encode('utf8')
            line = '%s\t?\n' % (word,)
            fp.write(line)
            #           print >> fp, "%s\t?" % (word,)

    chars, words, labels = load_pos_tagged_data("sample.conll", chardict, worddict, {'?': 0}, False)
    return [], chars, words, labels

def prepare_data(char_seqs, word_seqs, labels, maxlen, maxw, n_proj):
    """
    Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequences to this maximum
    length

    This swap the axis!
    """

    import pprint

    # x: a list of sentences
    lengths = [min(len(s_c), maxlen) for s_c in char_seqs]
    n_samples = len(char_seqs)

    x_c = numpy.zeros((maxlen, n_samples)).astype('int8')
    x_w = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    words_mask = numpy.zeros((maxw, maxlen, n_samples, n_proj)).astype(theano.config.floatX)
    y = numpy.zeros((maxw, n_samples)).astype('int8')
    y_mask = numpy.zeros((maxw, n_samples)).astype('int8')


    for idx, (s_c, s_w, l) in enumerate(zip(char_seqs, word_seqs, labels)):
        # idx is the current position in the mini-batch
        # s_c is a list of characters
        # s_w is a list of words
        # l is a list of labels
        s_c = s_c[:lengths[idx]]
        s_w = s_w[:lengths[idx]]
        l = l[:lengths[idx]]
        x_c[:lengths[idx], idx] = s_c
        x_w[:lengths[idx], idx] = s_w
        x_mask[:lengths[idx], idx] = 1.

        c = 0
        i = 0
        warning = None
        for j, a in enumerate(s_c):
            # j is the current character in this tweet
            # idx is the current tweet in this minibatch
            # c is the current word (can be up to 16)

            if a == 0:
                # This current character is a space
                # Increase the word count and continue
                c += 1
		if c >= 38:
			logging.warning("truncation")
			break
                continue

            words_mask[c, j, idx, :] = numpy.ones((n_proj,))

            y[c, idx] = l[c]
            y_mask[c, idx] = 1

        if warning is not None:
            logging.warning("%s", warning)

    return x_c, x_w, x_mask, words_mask, y, y_mask
