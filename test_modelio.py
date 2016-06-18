import unittest

import modelio

class ModelIOTests(unittest.TestCase):

    def test_char_dictionary_build(self):
        c = modelio.build_character_dictionary("Data/test_data.txt")
        self.assertEqual(c['s'], 1)
        self.assertEqual(c['o'], 2)
        self.assertEqual(c['z'], 23)

    def test_get_words(self):
        c = modelio.get_tweet_words("Data/test_data.txt")
        self.assertEqual(c[0][0], "some")
        self.assertEqual(c[3][4], "positivez")

if __name__ == "__main__":
    unittest.main()

