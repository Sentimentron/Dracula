'''
            e[:lengths[idx], idx] = l
Build a tweet sentiment analyzer
'''

from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from util import get_minibatches_idx, numpy_floatX
from modelio import load_pos_tagged_data, prepare_data

import imdb

datasets = {'imdb': (imdb.load_data, prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, wmask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, wmask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, wmask, y_mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, wmask, y_mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared', on_unused_input='warn')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, wmask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, wmask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    x.tag.test_value=numpy.asarray([[5, 5, 1], [5, 5, 1], [5, 5, 1]])
    mask = tensor.matrix('mask', dtype=config.floatX)
    mask.tag.test_value=numpy.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    wmask = tensor.matrix('wmask', dtype='int64')
    wmask.tag.test_value = numpy.random.randint(0, 13, (6, 4))
    y = tensor.matrix('y', dtype='int64')
    y.tag.test_value=numpy.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    y_mask = tensor.matrix('y_mask', dtype='int32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    # Mean pooling
    proj = proj * mask[:, :, None] # Remove any extraneous predictions

    avg_layer = tensor.alloc(numpy_floatX(0.), 16, n_samples, 128)
    count_layer = tensor.alloc(0, 16, n_samples, 128)
    fixed_ones  = tensor.ones_like(count_layer)

    #ydim = options[ydim]
    #y_default_tensor = tensor.allow(ydim,)
    #y_default_tensor = tensor.inc_subtensor(y_default_tensor[0], 1)

    def set_value_at_position(location, output_model, count_model, fixed_ones, values):
        print location.type, values.type, output_model.type
        output_subtensor = output_model[location[0], location[2]]
        count_subtensor = count_model[location[0], location[2]]
        ones_subtensor = fixed_ones[location[0], location[2]]
        values_subtensor = values[location[3], location[2]]
        return tensor.inc_subtensor(output_subtensor, values_subtensor), tensor.inc_subtensor(count_subtensor, ones_subtensor)

    (avg_layer, count_layer), _ = theano.foldl(fn=set_value_at_position,
                         sequences=[wmask],
                         outputs_info=[avg_layer, count_layer],
                         non_sequences=[fixed_ones, proj]
                         )

#   result = theano.printing.Print("RESULT", attrs=["shape"])(result)
#    avg_per_word = result.sum(axis=0, dtype=config.floatX).mean(axis=1)
#   avg_per_word = result.mean(axis=1, dtype=config.floatX)
#   avg_per_word = theano.printing.Print("AVG", attrs=["shape"])(avg_per_word)

    #avg_layer = theano.printing.Print("AVG_LAYER", attrs=['shape'])(avg_layer)
    count_layer_inverse = 1.0/count_layer
    count_layer_mask = 1.0 - tensor.isinf(count_layer_inverse)
    count_layer_mult = tensor.isinf(count_layer_inverse) + count_layer
    #count_layer_mult = theano.printing.Print("COUNT_LAYER_MULT")(count_layer_mult)
    #count_layer_mask = theano.printing.Print("COUNT_LAYER_MASK")(count_layer_mask)
    avg_per_word = (avg_layer / count_layer_mult) * count_layer_mask
    #avg_per_word = tensor.zeros_like(raw_avg_per_word)
    #avg_per_word = theano.printing.Print("AVG_PER_WORD")(avg_per_word)

#   proj = theano.printing.Print("PROJ")(proj)
#   avg_per_word = theano.printing.Print("AVG")(avg_per_word)

    #avg_per_word = tensor.set_subtensor(avg_per_word[y_mask.nonzero()], raw_avg_per_word[y_mask.nonzero()])

    print avg_per_word.type, proj.type

    #avg_per_word = theano.printing.Print("AVG_BEFORE")(avg_per_word)
    #y_mask = theano.printing.Print("YMASK", attrs=["shape"])(y_mask)
    #avg_per_word = tensor.set_subtensor(avg_per_word[y_mask[y_mask > 0].nonzero()], 0)
    #avg_per_word = theano.printing.Print("AVG_AFTER")(avg_per_word)

    #avg_per_word = tensor.dot(y_mask, avg_per_word)

    raw_pred, _ = theano.scan(fn=lambda p, free_variable: tensor.nnet.softmax(tensor.dot(p, tparams['U']) + tparams['b']),
                          outputs_info=None,
                          sequences=[avg_per_word, theano.tensor.arange(16)]
                          )

    #raw_pred = theano.printing.Print("PRED_BEFORE")(raw_pred)
    pred = tensor.zeros_like(raw_pred)
    pred = tensor.inc_subtensor(pred[:, :, 0], 1)
    #pred = tensor.set_subtensor(pred[y_mask[0, :], y_mask[1, :]], raw_pred[y_mask[0, :], y_mask[1, :]])
    pred = tensor.set_subtensor(pred[y_mask.nonzero()], raw_pred[y_mask.nonzero()])
    #pred = theano.printing.Print("PRED_AFTER", attrs=["shape"])(pred)

    # Ones where we don't care are set to zero
    # pred = tensor.dot(y_mask, pred)
    #pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])


    #pred = theano.printing.Print("PRED")(pred)
    f_pred_prob = theano.function([x, mask, wmask, y_mask], pred, name='f_pred_prob', on_unused_input='ignore')
    f_pred = theano.function([x, mask, wmask, y_mask], pred.argmax(axis=2), name='f_pred', on_unused_input='ignore')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    def cost_scan_i(i, j, free_var):
    #    i = theano.printing.Print("i", attrs=["shape"])(i)
    #    j = theano.printing.Print("j", attrs=["shape"])(j)
        return -tensor.log(i[tensor.arange(n_samples), j] + 1e-8)

#    cost, _ = theano.scan(fn=lambda i, j, free_variable: -tensor.log(i[tensor.arange(n_samples), j] + 1e-8),
 #                      outputs_info=None,
#                       sequences=[pred, y, tensor.arange(n_samples)])

    cost, _ = theano.scan(cost_scan_i, outputs_info=None, sequences=[pred, y, tensor.arange(n_samples)])

    cost = cost.mean()

    return use_noise, x, mask, wmask, y, y_mask, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, wmask, y, y_mask = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=140)
        pred_probs = f_pred_prob(x, mask, wmask, y_mask)
        #import pprint
        #pprint.pprint(pred_probs)
        #print pred_probs.shape, x.shape
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = []
    valid_shapes = []
    for _, valid_index in iterator:
        x, mask, wmask, y, y_mask = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=140)
        preds = f_pred(x, mask, wmask, y_mask)
        acc = numpy.equal(preds[y_mask.nonzero()], y[y_mask.nonzero()])
        valid_err.append(acc.sum())
        valid_shapes.append(numpy.count_nonzero(y_mask))

    valid_err = 1. - 1.0*numpy.asarray(valid_err).sum() / numpy.asarray(valid_shapes).sum()

    return valid_err

def split_at(src, prop):
    src_words, src_labels = [], []
    val_words, val_labels = [], []
    fin = max(int(prop * len(src[0])), 1)
    print len(src[0]), prop, fin
    for i, (l, p) in enumerate(zip(src[0], src[1])):
        if i < fin:
            val_words.append(l)
            val_labels.append(p)
        else:
            src_words.append(l)
            src_labels.append(p)
    return (src_words, src_labels), (val_words, val_labels)

def train_lstm(
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=20,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    load_data, prepare_data = get_dataset(dataset)

    # Load the training data
    print 'Loading data'
    #train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
    #                               maxlen=maxlen)
    char_dict = {}
    pos_dict = {}
    # Pre-populate the dictionaries
    load_pos_tagged_data("Data/TweeboOct27.conll", char_dict, pos_dict)
    load_pos_tagged_data("Data/TweeboDaily547.conll", char_dict, pos_dict)
    # Now load the data for real
    train = load_pos_tagged_data("Data/TweeboOct27.conll", char_dict, pos_dict)
    train, valid = split_at(train, 0.05)
    import pprint
    pprint.pprint(len(valid[0]))
    print len(train[0])
    test = load_pos_tagged_data("Data/TweeboDaily547.conll", char_dict, pos_dict)

    print len(test[0])

    #if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
    #    idx = numpy.arange(len(test[0]))
    #    numpy.random.shuffle(idx)
    #    idx = idx[:test_size]
    #    test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    print numpy.max(train[1], axis=None)
    ydim = numpy.max(numpy.amax(train[1])) + 1
    ydim = 26 # Hard-code, one that appears in the testing set, not in the training set

    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, wmask,
     y, y_mask, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, wmask, y, y_mask], cost, name='f_cost', on_unused_input='warn')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, mask, wmask, y, y_mask], grads, name='f_grad', on_unused_input='warn')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, wmask, y, y_mask, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.clock()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, wmask, y, y_mask = prepare_data(x, y)
                n_samples += x.shape[1]


                #print y
                cost = f_grad_shared(x, mask, wmask, y, y_mask)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', 100*(1-train_err), 'Valid ', 100*(1-valid_err),
                           'Test ', 100*(1-test_err))

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.clock()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    #pred_probs(f_pred_prob, prepare_data, test, kf_test)
    #sys.exit(1)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)


    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=100,
        test_size=500,
    )
