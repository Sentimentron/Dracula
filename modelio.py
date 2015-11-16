"""

	Contains I/O functions

"""

import numpy
import theano
import logging
import sys

from collections import defaultdict

def get_windowed(seq, window_length=16, overlap=8):
    # Base case: return whatever we have
    if len(seq) <= window_length:
        return [seq]

    # Otherwise, maintain a buffer of elements
    buf = []
    ret = []
    for idx, i in enumerate(seq):
        buf.append(i)
        if len(buf) == window_length:
            ret.append(tuple(buf))
            buf = buf[overlap+1:]

    if len(buf) != window_length - overlap - 1:
        ret.append(tuple(buf))

    return ret

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

def load_pos_tagged_data(path, chardict = {}, worddict={}, posdict={}, overlap=15, allow_append=True):

    if allow_append:
        build_character_dictionary(path, chardict)
        build_word_dictionary(path, worddict)
        build_tag_dictionary(path, posdict)

    words, chars, labels = [], [], []
    wordbuf, charbuf, labelsbuf = defaultdict(list), defaultdict(list), defaultdict(list)
    tweetidx = 0
    with open(path, 'r') as fin:
        for line in fin:
            cur_words, cur_chars, cur_labels = wordbuf[tweetidx], charbuf[tweetidx], labelsbuf[tweetidx]
            cur_word, cur_char, cur_label = [], [], []
            line = line.strip()
            if len(line) == 0:
                # Tweet boundary
                tweetidx += 1
                continue
            word, pos = line.split('\t')
            for c in '%s ' % (word, ):

                if c in chardict and c != ' ':
                    cur_char.append(chardict[c])
                elif c == ' ':
                    cur_char.append(0)
                else:
                    cur_char.append(0)

                if word in worddict:
                    cur_word.append(worddict[word])
                else:
                    cur_word.append(0)

            if pos in posdict:
                cur_label.append(posdict[pos])
            else:
                cur_label.append(0)

            cur_words.append(cur_word)
            cur_labels.append(cur_label)
            cur_chars.append(cur_char)

        for tweetidx in wordbuf:
            cur_words, cur_chars, cur_labels = wordbuf[tweetidx], charbuf[tweetidx], labelsbuf[tweetidx]
            for window in get_windowed(zip(cur_chars, cur_words, cur_labels), 16, overlap):
                window_chars, window_words, window_labels = [], [], []
                for (cs, ws, ls) in window:
                    for (c, w) in zip(cs, ws):
                        window_chars.append(c)
                        window_words.append(w)
                    for l in ls:
                        window_labels.append(l)
                words.append(window_words)
                chars.append(window_chars)
                labels.append(window_labels)
    return chars, words, labels

def string_to_unprepared_format(text, chardict, worddict):

    with open('sample.conll', 'wb') as fp:
        for word in text.split():
            print >> fp, "%s\t?" % (word,)

    chars, words, labels = load_pos_tagged_data("sample.conll", chardict, worddict, {'?': 0}, False)
    return [], chars, words, labels

    errors = []
    words, chars, labels = [], [], []
    cur_chars, cur_words, cur_labels = [], [], []
    for word in text.split():
        if word not in worddict:
            errors.append("%s not in word dictionary" % (word, ))
            wordidx = 0
        else:
            wordidx = worddict[word]

        for c in word:
            if c not in chardict:
                errors.append("%s not in char dictionary" % (c, ))
                charidx = 0
            else:
                charidx = chardict[c]

            chars.append(charidx)
            words.append(wordidx)
            labels.append(0)

        chars.append(0)
        words.append(wordidx)
        labels.append(0)

        cur_words.append(words)
        cur_chars.append(chars)
        cur_labels.append(labels)

        chars, words, labels = [], [], []

    return errors, cur_chars, cur_words, cur_labels

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
        warning = None
        for j, a in enumerate(s_c):
            # j is the current character in this tweet
            # idx is the current tweet in this minibatch
            # c is the current word (can be up to 16)

            if a == 0:
                # This current character is a space
                # Increase the word count and continue
                c += 1
                continue

            if c >= 16:
                # We can't represent more than 16 words per tweet
                # ATM
                warning = "Truncation! Words = %d" % (c,)
                continue

            if c >= len(l):
                warning = "Length mismatch! Words = %d, no labels = %d" % (c, len(labels))
                continue

            if j+1 >= maxlen:
                warning = "premature max-len break"
                break

            words_mask[j+1, idx] = c # Assign a nominal word for clarity
            # This provides the actual index into the LSTM output

            #print j, idx, c, maxlen, n_samples, 16
            #print numpy.ravel_multi_index((j, idx, c), (maxlen, n_samples, 16))

            words_mask[j+1, idx] = numpy.ravel_multi_index((j, idx, c), (maxlen, n_samples, 16))

            y[c, idx] = l[c]
            y_mask[c, idx] = 1

        if warning is not None:
            pass
            #logging.warning("%s", warning)

    return x_c, x_w, x_mask, words_mask, y, y_mask
