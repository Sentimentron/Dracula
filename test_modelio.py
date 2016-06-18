import unittest

import modelio

class ModelIOTests(unittest.TestCase):

    def test_char_dictionary_build(self):
        c = modelio.build_character_dictionary("Data/test_data.txt")
        self.assertEqual(c['s'], 1)
        self.assertEqual(c['o'], 2)
        self.assertEqual(c['z'], 23)

if __name__ == "__main__":
    unittest.main()

