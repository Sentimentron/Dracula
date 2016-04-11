#!/usr/bin/env python

from nltk.corpus import nps_chat

if __name__ == "__main__":

    for sentence in nps_chat.tagged_posts():
        for word, pos in sentence:
            print '{}\t{}'.format(word, pos)
        print
