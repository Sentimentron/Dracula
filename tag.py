#!/bin/env python

"""
Stores an individual pos-tagged word.
"""

class Tag(object):
    """Individual word-tag tuple"""
    def __init__(self, word, tag):
        """
        Creates a word-tag tuple
        :param word: The word to store
        :param tag: The associated part-of-speech tag
        """
        self.word = word
        self.tag = tag
