"""

	Contains I/O functions

"""

import numpy
import theano
import sys

def load_pos_tagged_data(path, chardict = {}, posdict={}):
    cur_words, cur_labels = [], []
    words, labels = [], []
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                words.append(cur_words[:-1])
                labels.append(cur_labels)
                cur_words = []
                cur_labels = []
                continue
            word, pos = line.split('\t')
            for c in word:
                if c not in chardict:
                    chardict[c] = len(chardict)+1
                cur_words.append(chardict[c])
                if pos not in posdict:
                    posdict[pos] = len(posdict)+1
            cur_labels.append(posdict[pos])
            cur_words.append(0)
    if len(cur_words) > 0:
    	words.append(cur_words)
    	labels.append(cur_labels)
    return words, labels

def prepare_data(seqs, labels, maxlen=None):
    """
    Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """

    import pprint

    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    #words_mask = numpy.zeros((maxlen, 80, n_samples)).astype(theano.config.floatX)
#   words_mask = numpy.zeros((maxlen * n_samples, 4)).astype('int64')
    words_mask = []
    y = numpy.zeros((16, n_samples)).astype('int64')
    y_mask = numpy.zeros((16, n_samples)).astype('int32')
    for idx, (s, l) in enumerate(zip(seqs, labels)):
        # idx is the current posdiction in the mini-batch
        # s is a list of characters
        # l is a list of labels
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

        c = 0
        i = 0
        for j, a in enumerate(s):
            if a == 0 or i >= 16:
                c += 1
                i = 1
                continue
            if c >= 16:
                # print >> sys.stderr, "Warning: truncation"
                break
            if c >= len(l):
                break
            # c is the current word
            # i is the current word index
            words_mask.append((c, i, idx, j))
#           print c, i, idx, j
#           words_mask[j + idx, 0] = c # First element stores the word index
#           words_mask[j + idx, 1] = i # Second stores the intra-word offset
#           words_mask[j + idx, 2] = idx # Original mini-batch
#           words_mask[j + idx, 3] = j # Original character index

            y[c, idx] = l[c]
            y_mask[c, idx] = 1
            i += 1

    words_mask = numpy.asarray(words_mask, dtype='int32')
#    print

    return x, x_mask, words_mask, y, y_mask