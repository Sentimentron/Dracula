"""

	Contains I/O functions

"""

import numpy
import theano
import sys

def load_pos_tagged_data(path, chardict = {}, worddict={}, posdict={}):
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

            if word not in worddict:
                worddict[word] = len(worddict)+1

            for c in word:
                if c not in chardict:
                    chardict[c] = len(chardict)+1
                cur_chars.append(chardict[c])
                cur_words.append(worddict[word])
                if pos not in posdict:
                    posdict[pos] = len(posdict)+1
            cur_labels.append(posdict[pos])
            cur_words.append(worddict[word])
            cur_chars.append(0)

    if len(cur_chars) > 0:
    	chars.append(cur_chars)
        words.append(cur_words)
    	labels.append(cur_labels)

    return chars, words, labels

def prepare_data(char_seqs, word_seqs, labels, maxlen=None):
    """
    Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """

    import pprint

    # x: a list of sentences
    lengths = [len(s_c) for s_c in char_seqs]

    if maxlen is not None:
        new_char_seqs = []
        new_word_seqs = []
        new_labels = []
        new_lengths = []
        for l, (s_c, s_w), y in zip(lengths, zip(char_seqs, word_seqs), labels):
            if l < maxlen:
                new_char_seqs.append(s_c)
                new_word_seqs.append(s_w)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        char_seqs = new_char_seqs
        word_seqs = new_word_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(lengths)
    maxlen = numpy.max(lengths)

    x_c = numpy.zeros((maxlen, n_samples)).astype('int8')
    x_w = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    words_mask = numpy.zeros((maxlen, n_samples), dtype='int32')
    y = numpy.zeros((16, n_samples)).astype('int8')
    y_mask = numpy.zeros((16, n_samples)).astype('int8')


    for idx, (s_c, s_w, l) in enumerate(zip(char_seqs, word_seqs, labels)):
        # idx is the current position in the mini-batch
        # s_c is a list of characters
        # s_w is a list of words
        # l is a list of labels
        x_c[:lengths[idx], idx] = s_c
        x_w[:lengths[idx], idx] = s_w
        x_mask[:lengths[idx], idx] = 1.

    for idx, (s_c, s_w, l) in enumerate(zip(char_seqs, word_seqs, labels)):
        # idx is the current position in the mini-batch
        # s is a list of characters
        # l is a list of labels
        x_c[:lengths[idx], idx] = s_c
        x_w[:lengths[idx], idx] = s_w
        x_mask[:lengths[idx], idx] = 1.

        c = 0
        i = 0
        for j, a in enumerate(s_c):
            if a == 0 or i >= 16:
                c += 1
                i = 1
                continue
            if c >= 16:
                # print >> sys.stderr, "Warning: truncation"
                break
            if c >= len(l):
                break
            if j+1 >= maxlen:
                break
            words_mask[j+1, idx] = c # Assign a nominal word for clarity
            # This provides the actual index into the LSTM output

            words_mask[j+1, idx] = numpy.ravel_multi_index((j, i, c), (maxlen, n_samples, 16))

            y[c, idx] = l[c]
            y_mask[c, idx] = 1
            i += 1

    return x_c, x_w, x_mask, words_mask, y, y_mask
