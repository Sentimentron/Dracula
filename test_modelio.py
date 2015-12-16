from modelio import *

import unittest

class TestModelIOMethods(unittest.TestCase):

    def test_build_character_dictionary(self):

        char = build_character_dictionary("Data/test_read_1.conll")
        self.assertEquals(char['R'], 1)
        self.assertEquals(char['T'], 2)
        self.assertEquals(char['@'], 3)
        self.assertEquals(char['d'], 4)
        self.assertEquals(char['l'], 5)
        self.assertEquals(char['o'], 6)
        self.assertEquals(char['v'], 7)
        self.assertEquals(char['a'], 8)
        self.assertEquals(char['t'], 9)
        self.assertEquals(char[':'], 10)
        self.assertEquals(char['j'], 11)
        self.assertEquals(char['e'], 12)
        self.assertEquals(char['n'], 13)
        self.assertEquals(char['s'], 14)
        self.assertEquals(char['h'], 15)
        self.assertEquals(char[','], 16)
        self.assertEquals(char['y'], 17)
        self.assertEquals(char['H'], 18)
        self.assertEquals(char['A'], 19)
        self.assertEquals(char['N'], 20)
        self.assertEquals(char['K'], 21)
        self.assertEquals(char['S'], 22)
        self.assertEquals(char['r'], 23)
        self.assertEquals(char['k'], 24)
        self.assertEquals(char['!'], 25)
        self.assertEquals(len(char), 25)

    def test_build_word_dictionary(self):

        words = build_word_dictionary("Data/test_read_1.conll")

        self.assertEquals(words["RT"], 1)
        self.assertEquals(words["@ddlovato"], 2)
        self.assertEquals(words[":"], 3)
        self.assertEquals(words["@joejonas"], 4)
        self.assertEquals(words["oh"], 5)
        self.assertEquals(words[","], 6)
        self.assertEquals(words["hey"], 7)
        self.assertEquals(words["THANKS"], 8)
        self.assertEquals(words["jerk"], 9)
        self.assertEquals(words["!"], 10)

    def test_build_tag_dictionary(self):

        pos = build_tag_dictionary("Data/test_read_1.conll")

        self.assertEquals(pos['~'], 1)
        self.assertEquals(pos['@'], 2)
        self.assertEquals(pos['!'], 3)
        self.assertEquals(pos[','], 4)
        self.assertEquals(pos['N'], 5)
        self.assertEquals(len(pos), 5)

    def test_get_maximum_tweet_length(self):

        max = get_max_tweet_length("Data/test_read_1.conll")
        self.assertEquals(max, 38)

    def test_get_numbers_of_tweets(self):

        len = get_number_of_tweets("Data/test_read_1.conll")
        self.assertEquals(len, 1)

    def test_get_max_words_in_tweet(self):

        maxlen = get_max_words_in_tweet("Data/test_read_1.conll")
        self.assertEquals(maxlen, 10)

    def test_get_maximum_word_offset(self):
        maxoff = get_max_word_offset("Data/test_read_1.conll")

        self.assertEquals(maxoff, 10)


    def test_read_data(self):

        char = build_character_dictionary("Data/test_read_1.conll")
        word = build_word_dictionary("Data/test_read_1.conll")
        pos = build_tag_dictionary("Data/test_read_1.conll")

        Wmask, chars, words, tags = read_data("Data/test_read_1.conll", 2, char, word, pos)
        self.assertEquals(Wmask.shape, (11, 39, 1, 2))
        self.assertEquals(chars.shape, (39, 1))
        self.assertEquals(words.shape, (39, 1))

        print chars

        self.assertEquals(chars[1, 0], char['R'])
        self.assertEquals(chars[2, 0], char['T'])
        self.assertEquals(chars[3, 0], char['@'])

        self.assertEquals(words[1, 0], word['RT'])
        self.assertEquals(words[3, 0], word['@ddlovato'])

        self.assertEquals(tags[1, 0], pos['~'])
        self.assertEquals(tags[3, 0], pos['@'])

        print Wmask[1, 1, :]
        self.assertEquals(Wmask[1, 1, 0, 0], 1.0)
        self.assertEquals(Wmask[1, 1, 0, 1], 1.0)
        self.assertEquals(Wmask[2, 1, 0, 0], 0.0)
        self.assertEquals(Wmask[2, 3, 0, 0], 1.0)