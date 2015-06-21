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
