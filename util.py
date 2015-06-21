#!/bin/env python
"""
    Utility functions
"""
import requests
import logging
import os

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
    files = [f for f in tmp if os.path.isfile(f)]
    extens = [f for f in files if os.path.splitext(f)[1] == "."+extension]
    return extens
