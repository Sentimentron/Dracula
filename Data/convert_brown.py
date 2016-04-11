with open("Brown-Retagged.txt") as fp:
    for line in fp:
        line = line.strip().split()
        for tuple in line:
            word, _, pos = tuple.partition('_')
            print '{}\t{}'.format(word, pos)
        print ''