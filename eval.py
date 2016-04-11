#!/bin/env python

# Seperate evaluation script

import sys
import csv
import requests
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter

tag_dict = {}
inv_tag_dict = {}
all_tags_ref = []
all_tags_resp = []
wrong_dict = Counter()

documents_received = 0
documents_all_correct = 0

debug_dump = open('debug.csv', 'w')

def get_from_server(words, tags, debugf):
    global tag_dict
    global inv_tag_dict
    global all_tags_ref
    global all_tags_resp
    global wrong_dict

    global documents_received
    global documents_all_correct

    payload = {'text': ' '.join(words)}
    r = requests.get("http://localhost:5000/api/tag", params=payload)
    print r.content
    response = r.json()

    print len(response["tags"]), len(tags)
    print tags
    print response["tags"]
    assert len(response["tags"]) == len(tags)
    text = response["text"].split()
    documents_received += 1
    if tags == response["tags"]:
       print "ALL SAME"
       documents_all_correct += 1
    else:
       print "DIFFER"
    for i, (r, s) in enumerate(zip(tags, response["tags"])):
        all_tags_ref.append(tag_dict[r])
        if s not in tag_dict:
            tag_dict[s] = len(tag_dict)
            inv_tag_dict[tag_dict[s]] = s
        all_tags_resp.append(tag_dict[s])
        print >> debug_dump, ",".join([text[i], r, s])
        debugf.writerow([text[i], r, s])
        if r != s:
            wrong_dict.update([(r, s)])
            try:
                print "WRONG: '%s' %s should be %s (times wrong %d)" % (text[i].encode('utf8'), s, r, wrong_dict[(r, s)])
            except:
                print "WRONG: %s should be %s (times wrong %d)" % (s, r, wrong_dict[(r, s)])

    print("Accuracy %.4f", accuracy_score(all_tags_ref, all_tags_resp))
    print("All correct: %.4f", 100.0 * documents_all_correct /
    documents_received)


with open(sys.argv[1]) as fin:
    with open("debug_dump.csv", "wb") as csvfile:
        # Read the file in CONLL format
        debugwriter = csv.writer(csvfile, delimiter=',',
                                 quotechar='"', quoting=csv.QUOTE_MINIMAL)
        debugwriter.writerow(["word","reference","tagged_as"])
        words, lengths, tags = [], [], []
        for line in fin:
            line = line.decode('utf8').strip()
            if len(line) == 0:
                # Update the results with what comes back
                get_from_server(words, tags, debugwriter)
                words, tags = [], []
                lengths = []
                continue
            try:
                word, tag = line.split()
                if tag not in tag_dict:
                    tag_dict[tag] = len(tag_dict)
                    inv_tag_dict[tag_dict[tag]] = tag

                if len(word) + sum(lengths) <= 560:
                    words.append(word)
                    tags.append(tag)
                    lengths.append(len(words))
            except ValueError:
                    continue

        assert len(words) == 0

names = [inv_tag_dict[k] for k in sorted(inv_tag_dict)]
print ",".join(names)
C=confusion_matrix(all_tags_ref, all_tags_resp)
import numpy
numpy.savetxt("confusion.csv", C, delimiter=",")
print C
print(classification_report(all_tags_ref, all_tags_resp, target_names=names))
print("Accuracy %.4f", accuracy_score(all_tags_ref, all_tags_resp))

