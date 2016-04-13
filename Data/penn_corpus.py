__author__ = 'rtownsend'

from nltk.corpus import treebank

for sentence in treebank.tagged_sents():
    for word, pos in sentence:
        if 'NONE' in pos:
            continue
        print '{}\t{}'.format(word, pos)
    print ''