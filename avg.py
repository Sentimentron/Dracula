#!/usr/bin/env python
__author__ = 'rtownsend'

from util import list_files_with_extension
from tag import IDENT

import matplotlib.pyplot as plt
import pickle
import logging

import numpy as np

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.DEBUG)
    # Data
    X = [[] for _ in range(len(IDENT))]
    labels = [x for x in IDENT]

    for f in list_files_with_extension("Data", "pkl"):
        logging.info("Loading %s...", f)
        with open(f, 'r') as fin:
            data = pickle.load(fin)
            for t in data:
                X[t.tag].append(len(t.word))

    # Plot
    plt.figure()
    for label, x in zip(labels, X):
        if len(x) == 0:
            continue
        print x
        plt.hist(x, bins=20, histtype='stepfilled', label=label)

    plt.title('POS-tag/length histogram')
    plt.xlabel('Length')
    plt.ylabel('probability')
    plt.legend()
    plt.show()

