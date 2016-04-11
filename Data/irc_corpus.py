#!/usr/bin/env python

from nltk.corpus import nps_chat

if __name__ == "__main__":

    for sentence in nps_chat.tagged_posts():
        for word, pos in sentence:
            if pos == '^PRP^VBP':
                pos = 'PRP$'
            if '^' in pos:
                pos = pos.replace('^', '')
            if pos == 'BES':
                pos = 'VBZ'
            if pos == 'HVS':
                pos = 'VBZ'
            if pos == 'X':
                pos = 'UH'
            if pos == 'GW':
                pos = 'PRP'
            if pos == 'LS':
                pos = 'JJS'
            print '{}\t{}'.format(word, pos)
        print
