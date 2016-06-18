#!/usr/bin/python
# loads outputs of ritter + gimpel tagger, reports on agreement
# corpusdiff.py ritter-file gimpel-file matches-output-file

import re
import sys

ptb_tags={}
ptb_tags['N'] = ['NN', 'NNS']
ptb_tags['O'] = ['PRP', 'WP']
ptb_tags['S'] = []
ptb_tags['^'] = ['NNP', 'NNPS']
ptb_tags['Z'] = ['POS']
ptb_tags['L'] = []
ptb_tags['M'] = []
ptb_tags['V'] = ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
ptb_tags['A'] = ['JJ', 'JJR', 'JJS']
ptb_tags['R'] = ['WRB', 'RB', 'RBR', 'RBS']
ptb_tags['!'] = ['UH']
ptb_tags['D'] = ['WDT', 'DT', 'WP$', 'PRP$']
ptb_tags['P'] = ['IN', 'TO']
ptb_tags['&'] = ['CC']
ptb_tags['T'] = ['RP']
ptb_tags['X'] = ['EX', 'PDT']
ptb_tags['Y'] = []
ptb_tags['#'] = ['HT']
ptb_tags['@'] = ['USR']
ptb_tags['~'] = ['RT', ':']
ptb_tags['U'] = ['URL']
ptb_tags['E'] = ['UH']
ptb_tags['$'] = ['CD']
ptb_tags[','] = ['#', '$', '.', ',', ':', '(', ')', '"', "'", "''", "``", 'UH']
ptb_tags['G'] = ['FW', 'POS', 'SYM', 'LS']

all_ptb = [val for subl in ptb_tags.values() for val in subl]
print all_ptb

# open both files
ritter_lines = open(sys.argv[1], 'r').read().strip().split('\n')
gimpel_lines = open(sys.argv[2], 'r').read().strip().split('\n')
matches = open(sys.argv[3], 'w')

if len(ritter_lines) != len(gimpel_lines):
    print 'ritter:', len(ritter_lines)
    print 'gimpel:', len(gimpel_lines)
    sys.exit('unequal number of returns in the two files')

# go through line by line
for i in range(len(ritter_lines)):
    print '->'

# reset flat
    sentences_match = True
    
    ritter_tokens = ritter_lines[i].strip().split(' ')
    gimpel_tokens = gimpel_lines[i].strip().split(' ')
    
    if len(ritter_tokens) != len(gimpel_tokens):
        print 'unequal number of tokens on line', i
        print 'ritter:', ritter_tokens
        print 'gimpel:', gimpel_tokens
        sys.exit()
    
# compare token by token
    for j in range(len(ritter_tokens)):

# do hashtag, username, url processing
        if ritter_tokens[j][0] == '#':
            ritter_tokens[j] = '_'.join(ritter_tokens[j].split('_')[:-1] + ['HT'])
            print 'hashtag found', ritter_tokens[j]
            continue

        if ritter_tokens[j][0] == '@':
            ritter_tokens[j] = '_'.join(ritter_tokens[j].split('_')[:-1] + ['USR'])
            print 'username found', ritter_tokens[j]
            continue

# fix ritter punc tag bug
        if re.match(r'^\.+_$', ritter_tokens[j]):
            ritter_tokens[j] += ':'
            print 'repaired', ritter_tokens[j], ritter_tokens
        if re.match(r'^\-+_$', ritter_tokens[j]):
            ritter_tokens[j] += ':'
            print 'repaired', ritter_tokens[j], ritter_tokens
        if ritter_tokens[j] in (':_', ';_'):
            ritter_tokens[j] += ':'
            print 'repaired', ritter_tokens[j], ritter_tokens

# check for missing labels 
        if ritter_tokens[j][-1] == '_':
            sentences_match=False
            print 'missing label on', ritter_tokens[j]
            break

# set flag if ritter ptbs don't fall within gimpel constraints
        ritter_tag = ritter_tokens[j].split('_')[-1]
        gimpel_tag = gimpel_tokens[j].split('_')[-1]
        
        if ritter_tag not in all_ptb:
            sentences_match = False
            print 'rit', ritter_tokens
            print 'tag '+ritter_tag+' not accounted for anywhere'
            break
        
        if ritter_tag not in ptb_tags[gimpel_tag]:
            sentences_match = False
            print 'Mismatch: '  
            print ritter_tag, 'not under', gimpel_tag
            print 'gim', ' '.join(gimpel_tokens)
            print 'rit', ' '.join(ritter_tokens)
            break

# if flag unset, print ritter line
    if sentences_match:
        print 'Match:'
        print ' '.join(ritter_tokens)
        matches.write(' '.join(ritter_tokens) + '\n')
