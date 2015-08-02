"""

	Contains I/O functions

"""

def load_pos_tagged_data(path, chardict = {}, posdict={}):
    cur_words, cur_labels = [], []
    words, labels = [], []
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                words.append(cur_words)
                labels.append(cur_labels)
                cur_words = []
                cur_labels = []
                continue
            word, pos = line.split('\t')
            for c in word:
                if c not in chardict:
                    chardict[c] = len(chardict)+1
                cur_words.append(chardict[c])
                if pos not in posdict:
                    posdict[pos] = len(posdict)+1
            cur_labels.append(posdict[pos])
            cur_words.append(0)
    if len(cur_words) > 0:
    	words.append(cur_words)
    	labels.append(cur_labels)
    return words, labels
