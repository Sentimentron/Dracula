#!/bin/env python
"""
    Utility functions
"""
import requests
import logging
import os
from theano import config
import numpy

def download_file(url, outpath):
    """
        Download a file to a given location over HTTP
    :param url: The HTTP location to fetch.
    :param outpath: The location to save it to
    """
    # Don't bother re-downloading stuff
    if os.path.exists(outpath):
        logging.info("%s already exists, skipping...", outpath)
        return

    logging.info("Downloading %s => '%s'", url, outpath)
    r = requests.get(url)
    with open(outpath, 'w') as fout:
        fout.write(r.content)

def list_files_with_extension(directory, extension):
    """
    Return a list of files with a given extension.
    :param directory: The directory to explore.
    :param extension: The extension to filter on.
    :return: The list of matching files.
    """
    tmp = os.listdir(directory)
    logging.debug(tmp)
    extens = [f for f in tmp if os.path.splitext(f)[1] == "."+extension]
    return [os.path.join(directory, f) for f in extens]

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def numpy_floatX(data):
    """
        Returns the source as an appropriate float type.
    """
    return numpy.asarray(data, dtype=config.floatX)

