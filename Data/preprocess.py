import io
import random
from twokenize import tokenizeRawTweetText

with open("quora_duplicate_questions.tsv", mode='r', encoding='utf8') as fin:
    ids = {}
    positives = []
    negatives = []
    for c, line in enumerate(fin):
        if c == 0:
            continue
        line = line.strip().split('\t')
        if len(line) <= 5:
            raise ValueError(line)
        line[3] = " ".join(tokenizeRawTweetText(line[3]))
        line[4] = " ".join(tokenizeRawTweetText(line[4]))
        line[3] = '"' + line[3].encode('unicode-escape').decode('utf8') + '"'
        line[4] = '"' + line[4].encode('unicode-escape').decode('utf8') + '"'
        ids[line[0]] = line
        if line[5] == "1":
            positives.append(line[0])
        elif line[5] == "0":
            negatives.append(line[0])
        else:
            raise ValueError(line[5])

    m = min(len(positives), len(negatives))
    random.shuffle(positives)
    random.shuffle(negatives)
    positives = positives[:m]
    negatives = negatives[:m]

    out = positives + negatives
    random.shuffle(out)

    for _id in out:
        line = ids[_id]
        print('\t'.join(line))
