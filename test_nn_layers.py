import unittest
import theano
import theano.tensor
import numpy
import logging

"""from nn_layers import PerWordAveragingOp


class WordAveragingOpTests(unittest.TestCase):

	def setup(self):
		self.op = PerWordAveragingOp()

	def test_grad(self):

		n_samples, dim = 16, 16
		proj = numpy.random.rand(16, 16)
		wmask = [[1, 2, 3, 4]]

		wmask = theano.tensor.matrix()
		proj = theano.tensor.tensor3()
		n_samples = theano.tensor.scalar()
		dim = theano.tensor.scalar()

		f = theano.function([proj, wmask, n_samples, dim], PerWordAveragingOp()(proj, wmask, n_samples, dim))

		op = PerWordAveragingOp()(proj, wmask, n_samples, dim)

		theano.tests.unittest_tools.verify_grad(f,
												[numpy.random.rand(16, 16, 16)])


"""

from nn_layers import per_word_averaging_layer

class WordAveragingOpTests(unittest.TestCase):

	"""def test_forward(self):

		n_samples, n_proj, n_chars = 2, 4, 4
		# n_chars -> maximum character sequence length (e.g. 140)
		# n_samples = minibatch size
		# n_proj = number of combined word/character embeddings
		n_chars, n_samples, n_proj = 4, 2, 4
		# Allocate a matrix which represents the LSTM layer output
		L = numpy.zeros((n_chars, n_samples, n_proj))
		L[0, 0, :] = [0.2, 0.3, 0.4, 0.5]
		L[1, 0, :] = [-0.2, -0.4, 0.6, 0.7]
		L[2, 0, :] = [0.8, -0.2, 0.3, 0.1]
		L[0, 1, :] = [0.4, 0.2, 0.8, 0.9]

		W = numpy.zeros((4, 3), dtype='int32')
		W[0, :] = [1, 0, 1] # First word, first tweet, first character
		W[1, :] = [1, 0, 2]	# First word, first tweet, second character
		W[2, :] = [2, 0, 3] # Second word, first tweet, third character
		W[3, :] = [1, 1, 1] # First word, second tweet, first character

		# The expected output
		O = numpy.zeros((16, n_samples, n_proj))
		O[0, 0, :] = [0.0, -0.05, 0.50, 0.60] # First word, first tweet
		O[1, 0, :] = [0.8, -0.2, 0.3, -0.1]   # Second word, first tweet
		O[0, 1, :] = [-0.4, 0.2, 0.8, 0.9]    # First word, second tweet

		LArg = theano.tensor.dtensor3()
		WArg = theano.tensor.imatrix()
		#nSamplesArg = theano.tensor.scalar()
		#dimArg = theano.tensor.scalar()

		out = per_word_averaging_layer(LArg, WArg, n_samples, n_proj)

		f = theano.function([LArg, WArg], out)

		O_actual = f(L, W)

		print O
		print O_actual

		#print O[0, 0, :] - O_actual[0, 0, :]

		self.assertTrue(numpy.allclose(O, O_actual))"""


	def test_numpy(self):
		n_chars, n_samples, n_proj = 4, 2, 4
		# Allocate a matrix which represents the LSTM layer output
		L = numpy.zeros((n_chars, n_samples, n_proj))
		L[1, 0, :] = [0.2, 0.3, 0.4, 0.5]
		L[2, 0, :] = [-0.2, -0.4, 0.6, 0.7]
		L[3, 0, :] = [0.8, -0.2, 0.3, -0.1]
		L[1, 1, :] = [-0.4, 0.2, 0.8, 0.9]

		W = numpy.zeros((n_chars, n_samples), dtype='int32')
		W[1, 0] = 1 # First tweet, first character to first word
		W[2, 0] = 1 # First tweet, second character to second word
		W[3, 0] = 2 # First tweet, third character to second word
		W[1, 1] = 1 # Second tweet, first character to second word

		# The expected output
		O = numpy.zeros((n_samples, 16, n_proj))
		O[0, 1, :] = [0.0, -0.05, 0.50, 0.60] # First word, first tweet
		O[0, 2, :] = [0.8, -0.2, 0.3, -0.1]   # Second word, first tweet
		O[1, 1, :] = [-0.4, 0.2, 0.8, 0.9]    # First word, second tweet

		# First index is the word index
		# Second is the character
		# Third is the batch index
		tmp = numpy.zeros((n_chars, n_samples, 16, n_proj))
		for i in range(n_chars):
			for j in range(n_samples):
				tmp[i, j, W[i,j], :] += L[i, j, :]

		divider = (tmp != 0).sum(axis=0)
		divider += (divider == 0.) # Make sure we don't get NaN
		O_actual = tmp.sum(axis=0) / divider

		self.assertTrue(numpy.allclose(O_actual, O))


	def test_numpy_2(self):
		n_chars, n_samples, n_proj = 4, 2, 4
		# Allocate a matrix which represents the LSTM layer output
		L = numpy.zeros((n_chars, n_samples, n_proj))
		L[1, 0, :] = [0.2, 0.3, 0.4, 0.5]
		L[2, 0, :] = [-0.2, -0.4, 0.6, 0.7]
		L[3, 0, :] = [0.8, -0.2, 0.3, -0.1]
		L[1, 1, :] = [-0.4, 0.2, 0.8, 0.9]

		W = numpy.zeros((n_chars, n_samples), dtype='int32')
		W[1, 0] = 1 # First tweet, first character to first word
		W[2, 0] = 1 # First tweet, second character to second word
		W[3, 0] = 2 # First tweet, third character to second word
		W[1, 1] = 1 # Second tweet, first character to second word

		Wref = numpy.zeros(W.shape, dtype='int32')
		Wref[:, :] = W[:, :]
		for i in range(n_chars):
			for j in range(n_samples):
				print i, j, W[i, j]
				W[i, j] = ((i*n_chars + j)*n_samples)* W[i,j]

				W[i, j] = numpy.ravel_multi_index((i, j, Wref[i, j]), (n_chars, n_samples, 16))


		logging.debug(len(W.flatten()))
		# The expected output
		O = numpy.zeros((n_samples, 16, n_proj))
		O[0, 1, :] = [0.0, -0.05, 0.50, 0.60] # First word, first tweet
		O[0, 2, :] = [0.8, -0.2, 0.3, -0.1]   # Second word, first tweet
		O[1, 1, :] = [-0.4, 0.2, 0.8, 0.9]    # First word, second tweet

		# First index is the word index
		# Second is the character
		# Third is the batch index
		#tmp = np.reshape(L, (n_chars * n_samples, n_proj))[W.flatten()].reshape((n_chars, n_samples, 16, n_proj))
		tmpref = numpy.zeros((n_chars, n_samples, 16, n_proj))
		tmp = numpy.zeros((n_chars * n_samples * 16, n_proj))
		logging.debug(tmp.size)
		tmp[W.flatten()] = L.reshape((n_chars * n_samples, n_proj))
		logging.debug(tmp.size)
		tmp = numpy.reshape(tmp, (n_chars, n_samples, 16, n_proj))
		for i in range(n_chars):
			for j in range(n_samples):
				tmpref[i, j, Wref[i,j], :] += L[i, j, :]

		print tmp
		print "tmpRef"
		print tmpref
		divider = (tmp != 0).sum(axis=0)
		divider += (divider == 0.) # Make sure we don't get NaN
		O_actual = tmp.sum(axis=0) / divider

		self.assertTrue(numpy.allclose(O_actual, O))

if __name__ == "__main__":
	unittest.main()