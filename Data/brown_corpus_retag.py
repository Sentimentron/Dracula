__author__ = 'rtownsend'

"""
    Because we don't have access to the WSJ corpus, we'll have to use the
    Brown corpus but converted to Penn Treebank tags.
"""

import argparse
from nltk.corpus import brown
import subprocess
import tempfile

if __name__ == "__main__":

    p = argparse.ArgumentParser("Re-tag the Brown corpus with Penn Treebank tags")
    p.add_argument("tagger", help="Path to a competent tagger")
    p.add_argument("model", help="Model file")

    a = p.parse_args()

    _, tmp = tempfile.mkstemp()

    buf = []
    with open(tmp, 'w') as f:
        for sentence in brown.tagged_sents():
            buf = []
            for word, _ in sentence:
                buf.append(word)
            f.write(' '.join(buf) + '\n')

    print subprocess.check_output(['java', '-jar', a.tagger, a.model, tmp])

