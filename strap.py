from mlp import MLP
import numpy

import theano
from theano import tensor as T

if __name__ == "__main__":
    rng = numpy.random.RandomState(1234)

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = MLP (
        rng=rng,
        input=x,
        n_in=2,
        n_hidden=5,
        n_out=2
    )

    data = numpy.asarray([[4., 3, 2], [1, 2, 1]])

    forward = theano.function(
        inputs=[index],
        outputs=classifier.pred(x),
        givens={
            x: data[index:index+2]
        }
    )
    cost = -T.log(classifier.pred[y] + 1e-8).mean() + 0.00 * classifier.L1 + 0.00 * classifier.L2_sqr

    print forward(0)