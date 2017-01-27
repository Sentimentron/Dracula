import io
from twokenize import tokenizeRawTweetText

with open("quora_duplicate_questions.tsv", mode='r', encoding='utf8') as fin:
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
        print('\t'.join(line))
