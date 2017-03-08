#!/usr/bin/env python3

import argparse
import csv
import unicodedata
from twokenize import tokenizeRawTweetText

if __name__ == "__main__":
    a = argparse.ArgumentParser("Pre-process and reformat training data")
    a.add_argument("file", help="The file to process")
    a.add_argument("output", help="The file to write to")

    p = a.parse_args()

    with open(p.output, "w") as fout:
        with open(p.file, "r", encoding='iso-8859-1') as fin:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            for row in reader:
                label = int(row[0])
                if label == 0:
                    label = -1
                elif label == 2:
                    label = 0
                elif label == 4:
                    label = 1
                else:
                    raise ValueError(label)
                text = row[-1]
                text = " ".join(tokenizeRawTweetText(text))
                text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
                text = text.decode('ascii')
                writer.writerow([text, label])



