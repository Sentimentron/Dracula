#!/bin/env python

# Seperate evaluation script

import sys
import csv
import requests
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter

if __name__ == "__main__":
    with open("Data/testing_data.txt", "rb") as fp:
        filereader = csv.reader(fp)
        Y, y = [], []
        for text, polarity in filereader:
            polarity = int(float(polarity)) + 1
            if polarity == 1:
                continue # Skip neutral
            payload = {'text': text}
            r = requests.get("http://localhost:5000/api/tag", params=payload)
            print r.content
            determined_polarity = r.json()["label"]
            Y.append(polarity)
            y.append(determined_polarity)
        print("Accuracy %.4f", accuracy_score(Y, y))
        print(classification_report(Y, y))
