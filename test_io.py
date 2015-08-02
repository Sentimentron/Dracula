"""

	Unit tests for I/O functions.

"""

import unittest

from io import load_pos_tagged_data

class TestIOMethods(unittest.TestCase):

	def test_load_pos_tagged_data_1(self):
		"""
		Check that the Tweebo POS format is read correctly.
		"""

		char = {}
		pos = {}
		words, labels = load_pos_tagged_data("Data/test_read_1.conll", char, pos)

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

		self.assertEquals(pos['~'], 1)
		self.assertEquals(pos['@'], 2)
		self.assertEquals(pos['!'], 3)
		self.assertEquals(pos[','], 4)
		self.assertEquals(pos['N'], 5)
		self.assertEquals(len(pos), 5)

		self.assertEquals(words[0][0:3], [1, 2, 0])
		self.assertEquals(words[0][3:13],  [3, 4, 4, 5, 6, 7, 8, 9, 6, 0])
		self.assertEquals(labels, [[1, 2, 1, 2, 3, 4, 3, 5, 5, 4]])

	def test_load_pos_tagged_data_2(self):
		"""
			Check we can handle a sample of 10 correctly.
		"""

		pass
