#!/bin/env python

"""
Stores an individual pos-tagged word.
"""

IDENT = {u'ADV': 0,
         u'NOUN': 1,
         u'ADP': 2,
         u'PRON': 3,
         u'DET': 4,
         u'.': 5,
         u'PRT': 6,
         u'VERB': 7,
         u'X': 8,
         u'NUM': 9,
         u'CONJ': 10,
         u'ADJ': 11}

class Tag(object):
    """Individual word-tag tuple"""
    def __init__(self, word, tag):
        """
        Creates a word-tag tuple
        :param word: The word to store
        :param tag: The associated part-of-speech tag
        """
        self.word = word
        self.tag = IDENT[tag]

    def get_readable_tag(self):
        """
        Returns what tag this is.
        :return: A human-readable tag
        """
        return IDENT.keys()[self.tag]

    def __str__(self):
        return "Tag(%s, %s)" % (self.word, self.get_readable_tag())
