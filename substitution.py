from modelio import load_pos_tagged_data
from matcher import MultiSimilarityMatcher
import logging
import pickle

"""

    This file generates a substitution dictionary for the test dataset.

    The idea here is that if you can substitute a word for something that's
    fairly similar, it's better than just relying on the zero word.

"""

logging.basicConfig(level=logging.DEBUG)

with open('substitutions.pkl', 'wb') as fout:

    threshold = 1

    word_dict, pop_dict = {}, {}
    load_pos_tagged_data("Data/Brown.conll", worddict=word_dict, popularity=pop_dict)
    load_pos_tagged_data("Data/TweeboOct27.conll", worddict=word_dict, popularity=pop_dict)

    for w in pop_dict:
        if pop_dict[w] <= threshold:
            word_dict.pop(w, None)

    test_dict, test_pop = {}, {}
    load_pos_tagged_data("Data/TweeboDaily547.conll", worddict=test_dict, popularity=test_pop)

    for w in test_pop:
        if test_pop[w] <= threshold:
            test_dict.pop(w, None)

    sim = MultiSimilarityMatcher()
    sim.update_from_dict(word_dict)

    sim.expand_dict(word_dict, test_dict)

    pickle.dump(word_dict, fout)
