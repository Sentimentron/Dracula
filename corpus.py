#!/bin/env python
"""
Download and pre-process the POS-tagged corpora that we will use.
"""
__author__ = 'rtownsend'

import nltk
import logging
from argparse import ArgumentParser
from tag import Tag
import cPickle as pickle
from util import download_file

TWEEBO_OCT27_DL = "https://raw.githubusercontent.com/brendano/ark-tweet-nlp/master/data/twpos-data-v0.3/oct27.conll"
TWEEBO_DAILY547_DL = "https://raw.githubusercontent.com/brendano/ark-tweet-nlp/master/data/twpos-data-v0.3/daily547.conll"

def process_cmd_line():
    """Read command line arguments"""
    p = ArgumentParser("Download and pre-process command line arguments")
    p.add_argument("--brown", action="store_true")
    p.add_argument("--tweebo", action="store_true")

    args = p.parse_args()

    if not args.brown and not args.tweebo:
        raise ValueError("No corpora specified (see --usage)")

    return args

def process_brown():
    """Read, convert and save the Brown corpus"""
    tags = []
    logging.info("Reading Brown corpus....")
    for word, tag in nltk.corpus.brown.tagged_words(tagset="universal"):
        t = Tag(word, tag)
        tags.append(t)
    logging.info("Saving....")
    with open("Data/brown.pkl", "w") as fout:
        pickle.dump(tags, fout, pickle.HIGHEST_PROTOCOL)

def process_tweebo():
    """Read, convert and save the Tweebo corpus"""
    download_file(TWEEBO_DAILY547_DL, "Data/TweeboDaily547.conll")
    download_file(TWEEBO_OCT27_DL, "Data/TweeboOct27.conll")

    def interpret_conll(path):
        """
        Read a CONLL file line-by-line and export the tags
        :param path: The path of the file
        :return: A list of Tag objects.
        """
        logging.info("Reading %s...", path)
        ret = []
        tweebo = ['N', 'O', '^', 'S', 'Z', 'V', 'L',
                  'M', 'A', 'R', '!', 'D', 'P', '&',
                  'T', 'X', 'Y', '#', '@', '~', 'U',
                  'E', '$', ',', 'G']
        ref = [u'NOUN', u'PRON', 'NOUN', 'DET', 'NOUN',
               'VERB', '']
        tagmap = {
            'N': u'NOUN',
            'O': u'PRON',
            '^': u'NOUN',
            'S': u'X',
            'Z': u'NOUN',
            'V': u'VERB',
            'L': u'PRON',
            'M': u'NOUN',
            'A': u'ADJ',
            'R': u'ADV',
            '!': u'.',
            'D': u'DET',
            'P': u'CONJ',
            '&': u'CONJ',
            'T': u'PRT',
            'X': u'DET',
            'Y': u'DET',
            '#': u'X',
            '@': u'NOUN',
            '~': u'X',
            'U': u'X',
            'E': u'.',
            '$': u'NUM',
            ',': u'.',
            'G': u'X'
        }
        with open(path, 'r') as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0:
                    continue
                line = line.split()
                word, raw = line
                t = Tag(word, tagmap[raw])
                ret.append(t)
        return ret

    d547 = interpret_conll("Data/TweeboDaily547.conll")
    o24 = interpret_conll("Data/TweeboOct27.conll")

    with open('Data/TweeboDaily547.pkl', 'w') as fout:
        logging.info("Saving daily...")
        pickle.dump(d547, fout, pickle.HIGHEST_PROTOCOL)

    with open('Data/TweeboOct27.pkl', 'w') as fout:
        logging.info("Saving Oct...")
        pickle.dump(o24, fout, pickle.HIGHEST_PROTOCOL)


def main():
    """
    Main method.
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.DEBUG)
    args = process_cmd_line()

    if args.brown:
        process_brown()
    if args.tweebo:
        process_tweebo()

if __name__ == "__main__":
    main()
