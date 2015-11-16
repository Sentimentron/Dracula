import unittest
import logging

from matcher import SimilarityMatcher, MultiSimilarityMatcher, char_diff
from modelio import load_pos_tagged_data

class MatcherTests(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)

    def test_matching(self):

        a = "ACACACTA"
        b = "AGCACACA"

        # So for whatever reason, the score matrix doesn't exactly match
        # the one provided by Wikipedia, chances are there's something wrong.
        score, sim = SimilarityMatcher._similarity(char_diff, -1.0, a, b)

        print score
        print sim

        self.assertEqual(12, score)

    def test_most_similar(self):

        word_dict = {}
        load_pos_tagged_data("Data/TweeboOct27.conll", worddict=word_dict)

        self.assertEquals(word_dict["I"], 1)

        sim = SimilarityMatcher()
        sim.update_from_dict(word_dict)

        score, word = sim.get_most_similar_word("iPod")

        self.assertEqual(word, "ipod")

    def test_expand_dictionary(self):

        word_dict = {}
        load_pos_tagged_data("Data/TweeboOct27.conll", worddict=word_dict)

        test_dict = {}
        load_pos_tagged_data("Data/TweeboDaily547.conll", worddict=test_dict)

        for w in list(word_dict):
            if w[0] != "i":
                word_dict.pop(w, None)

        for w in list(test_dict):
            if w[0] != "i":
                test_dict.pop(w, None)

        self.assertTrue("ipod" in word_dict)
        self.assertTrue("ipod" not in test_dict)
        self.assertTrue("iPod" not in word_dict)
        self.assertTrue("iPod" in test_dict)

        sim = MultiSimilarityMatcher()
        sim.update_from_dict(word_dict)

        sim.expand_dict(word_dict, test_dict)

        self.assertEqual(word_dict["ipod"], word_dict["iPod"])
