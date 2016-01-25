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
    p.add_argument("--tokens", nargs="+")

    args = p.parse_args()

    if not args.brown and not args.tokens:
        raise ValueError("No corpora specified (see --usage)")

    return args

#BROWN_TAG_DICT = {
#    'AT' : 'D', # Determiner, i.e. 'THE'
#    'NP-TL': 'Z', # Proper noun i.e. 'FULTON'
#    ''
#}

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

def process_brown():
    """
    Convert the Brown corpus tags into CONLL, substituting the
    native tags for something that would be roughly equivalent
    in the TweeboPOS corpus
    """

    def convert_tag(tag):
        """
            Converts a Brown corpus tag into a TweeboParse one.

        """
        if tag[0] == "*":
            # Negator, like NOT
            return "R"
        elif "AB" in tag:
            return "D"
        elif "AP" in tag:
            return "D"
        elif "AT" in tag:
            return "D"
        elif 'B' == tag[0]:
            return "V"
        elif tag[0:2] == "CC":
            return "&"
        elif tag[0:2] == "CS":
            return "P"
        elif tag[0:2] == "CD":
            return "$"
        elif tag == "CS":
            return "P"
        elif "DO" == tag[0:2]:
            return "V"
        elif "DT" == tag[0:2]:
            return "D"
        elif tag[0:2] == "EX":
            return "X"
        elif tag[0:2] == "FW":
            return "G"
        elif tag[0:2] == "HV":
            return "V"
        elif tag[0:2] == "IN": # Preposition
            return "P"
        elif tag[0:2] == "JJ":
            return "J"
        elif tag[0:2] == "MD":
            return "V"
        elif tag[0:2] == "NN":
            if tag == "NN+DNZ":
                return "S"
            elif tag == "NP+":
                return "Z"
            return "N"
        elif tag[0:2] == "NR":
            # Noun, singular, adverbial (e.g. downtown, Friday)
            return "^"
        elif tag[0:2] == "NP":
            return "^"
        elif tag[0:2] == "OP":
            return "$"
        elif tag[0:2] == "OD":
            return "$"
        elif tag[0:2] == "PN":
            if tag == "PN+":
                return "L"
            return "O"
        elif tag[0:3] == "PPL":
            return "O"
        elif tag[0:3] == "PPS":
            return "O"
        elif tag[0:3] == "PP$":
            return "O"
        elif tag[0:4] == "PP$$":
            return "O"
        elif tag == "PPS+":
            return "L"
        elif tag[0:3] == "PPO":
            return "O"
        elif tag[0:2] == "QL":
            return "R"
        elif tag[0:2] == "RB":
            # Adverb, comparative (e.g better)
            return "R"
        elif tag[0:2] == "RN":
            return "R"
        elif tag[0:2] == "RP":
            # Verb particle
            return "T"
        elif tag[0:2] == "TO":
            return "P"
        elif tag[0:2] == "UH":
            return "!"
        elif tag[0] == "V":
            return "V"
        elif tag[0:3] == "WDT":
            return "D"
        elif tag == "WQL":
            return "R"
        elif tag[0:2] == "WP":
            return "O"
        elif tag[0:3] == "WRB":
            return "R"
        elif tag in [",", "--", ".", ":", "(", ")", "''", '``', '\'']:
            return ","
        elif "," in tag:
            return ","
        elif "(" in tag:
            return ","
        elif ")" in tag:
            return ","
        elif ":" in tag:
            return ","
        elif "." in tag:
            return ","
        elif "-" in tag:
            return ","
        elif tag == "NIL":
            return "G"
        else:
            raise ValueError("Could not translate: " + tag)

    logging.info("Reading Brown corpus....")
    tags = set([])
    for sentence in nltk.corpus.brown.tagged_sents():
        for word, tag in sentence:
            if word == "this":
                print word, tag
            tags.add(tag)
            #print "%s\t%s" % (word, convert_tag(tag))
        #print ''

    print tags
    print len(tags)

def process_token(t):

    with open(t, 'r') as fin:
        for line in fin:
            tokens = line.split()
            for token in tokens:
                parts = token.split('_')
                word = '_'.join(parts[:-1])
                pos = parts[-1]
                print "%s\t%s" % (word, pos)
            print ""

def main():
    """
    Main method.
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.DEBUG)
    args = process_cmd_line()

    if args.brown:
        process_brown()

    if args.tokens:
        for t in args.tokens:
            process_token(t)


if __name__ == "__main__":
    main()
