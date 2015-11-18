__author__ = 'rtownsend'

import numpy
import logging

def char_diff(a, b):
    if a == b:
        return 2
    if a.lower() == b.lower():
        return 1 # It's a match
    return -1

class SimilarityMatcher(object):

    def __init__(self):
        self.words = set([])
        pass

    @classmethod
    def _similarity(cls, diff, gap, a, b):

        mat = numpy.zeros((len(b)+1, len(a)+1))

        for i in range(len(a)):
            for j in range(len(b)):


                d = max(0, mat[j, i] + diff(a[i], b[j]))
                d = max(d, mat[j + 1, i] + gap)
                d = max(d, mat[j, i + 1] + gap)
                mat [j + 1, i + 1] = d

        return numpy.max(mat), mat

    def get_most_similar_word(self, word):
        best_score, best_match = 0, ""
        for w in self.words:
            sim, _ = self._similarity(char_diff, -1, w, word)
            if sim > best_score:
                best_score = sim
                best_match = w

        return best_score, best_match

    def update_from_dict(self, d):
        self.words.update(d.keys())

class MultiSimilarityMatcher(object):

    def __init__(self):
        self.hashtags = SimilarityMatcher()
        self.words = SimilarityMatcher()
        self.mentions = SimilarityMatcher()

    def update_from_dict(self, d):
        words = {}
        mentions = {}
        tags = {}

        for w in d:
            if w[0] == '@':
                mentions[w] = True
            elif w[0] == '#':
                tags[w] = True
            else:
                words[w] = True

        self.hashtags.update_from_dict(tags)
        self.mentions.update_from_dict(mentions)
        self.words.update_from_dict(words)

    def get_most_similar_word(self, w):
        if w[0] == '@':
            return self.mentions.get_most_similar_word(w)
        elif w[0] == '#':
            return self.hashtags.get_most_similar_word(w)
        else:
            return self.words.get_most_similar_word(w)

    def expand_dict(self, target, other):
        self.update_from_dict(target)

        words_to_find = set([])
        for w in other:
            if w not in target:
                words_to_find.add(w)

        logging.info("We've got %d words to find", len(words_to_find))

        for w in words_to_find:
            _, m = self.get_most_similar_word(w)
            target[w] = target[m]
            logging.debug("'%s' => '%s'", w, m)
