'''
Build a tweet sentiment analyzer
'''

from argparse import ArgumentParser
import logging

import itertools

import cPickle as pkl
import time

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

from util import get_minibatches_idx
from modelio import load_pos_tagged_data, prepare_data, get_max_word_count, get_max_length, get_max_word_length

from nn_layers import *
from nn_lstm import lstm_layer, lstm_unmasked_layer, bidirectional_lstm_layer
from nn_params import *
from nn_optimizers import *
from nn_support import pred_error
from nn_serialization import zipp, unzip, load_params

import os.path
import pickle

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def build_model(tparams, options, maxw, training=True):
    xc = tensor.tensor3('xc', dtype='int8')
    mask = tensor.tensor4('mask', dtype=config.floatX)
    y = tensor.matrix('y', dtype='int8')
    y_mask = tensor.matrix('y_mask', dtype='int8')

    n_batch = xc.shape[2]

    emb = embeddings_layer(xc, tparams['Cemb'], options['dim_proj'])

    dist = emb
    dist_mask = mask

    dist = dist * dist_mask

    for i in range(options['letter_layers']):
        name = 'lstm_chars_%d' % (i + 1,)

        def _step(x_):
            t = bidirectional_lstm_layer(tparams, x_, options, name)
            return t

        dist, updates = theano.scan(_step, sequences=[dist], n_steps=dist.shape[0])

    proj2 = per_word_averaging_layer(dist, dist_mask)

    for i in range(options['word_layers']):
        name = 'lstm_words_%d' % (i + 1,)
        proj2 = bidirectional_lstm_layer(tparams, proj2, options, name)

    pred = softmax_layer(proj2, tparams['U'], tparams['b'], y_mask, maxw, training)
    #pred = theano.printing.Print("pred", attrs=["shape"])(pred)

    f_pred_prob = theano.function([xc, mask, y_mask], pred, name='f_pred_prob', on_unused_input='ignore')
    f_pred = theano.function([xc, mask, y_mask], pred.argmax(axis=2), name='f_pred', on_unused_input='ignore')

    def cost_scan_i(i, j, free_var):
        return -tensor.log(i[tensor.arange(n_batch), j] + 1e-8)

    cost, _ = theano.scan(cost_scan_i, outputs_info=None, sequences=[pred, y, tensor.arange(n_batch)])

    cost = cost.mean()

    return xc, mask, y, y_mask, f_pred_prob, f_pred, cost

def split_at(src, prop):
    src_chars, src_words, src_labels = [], [], []
    val_chars, val_words, val_labels = [], [], []
    fin = max(int(prop * len(src[0])), 1)
    print len(src[0]), prop, fin
    for i, ((c, w), l) in enumerate(zip(zip(src[0], src[1]), src[2])):
        if i < fin:
            val_chars.append(c)
            val_words.append(w)
            val_labels.append(l)
        else:
            src_chars.append(c)
            src_words.append(w)
            src_labels.append(l)
    return (src_chars, src_words, src_labels), (val_chars, val_words, val_labels)

def train_lstm(
    dim_proj_chars=32,  # character embedding dimension and LSTM number of hidden units.
    patience=4,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.0001,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=900,  # Compute the validation error after this number of update.
    saveFreq=2220,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=50,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    pretrain = False, # If True, load some data from this argument
    # Use to keep track of feature enumeration
    char_dict = {},
    word_dict = {},
    pos_dict = {},
    word_layers=0,
    letter_layers=0
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    if reload_model is not None:
        load_params(reload_model, model_options)
        char_dict = model_options['char_dict']
        word_dict = model_options['word_dict']
        pos_dict = model_options['pos_dict']

    # Load the training data
    print 'Loading data'

    input_path = "Data/Gate-Train.conll"

    load_pos_tagged_data(input_path, char_dict, word_dict, pos_dict)

    with open("substitutions.pkl", "rb") as fin:
        word_dict = pickle.load(fin)

    max_word_count = 0
    max_word_length = 0
    max_length = 0
    # Now load the data for real
    data = load_pos_tagged_data(input_path, char_dict, word_dict, pos_dict, 0)
    train, eval = split_at(data, 0.05)
    test, valid = split_at(eval, 0.50)
    max_word_count = max(max_word_count, \
    get_max_word_count(input_path))
    max_word_length = max(max_word_length, \
    get_max_word_length(input_path))
    max_length = max(max_word_length, get_max_length(input_path))

    #ydim = numpy.max(numpy.max(train[2])) + 1
    #print numpy.max(train[2])
    ydim = max(itertools.chain.from_iterable(train[2])) + 1
    print "ydim =", ydim

    model_options['ydim'] = ydim
    model_options['n_chars'] = len(char_dict)+1
    model_options['n_words'] = len(word_dict)+1

    model_options['word_dict'] = word_dict
    model_options['char_dict'] = char_dict
    model_options['pos_dict'] = pos_dict

    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options, reload_model is not None)

    logging.info('Building model')

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (xc, mask,
     y, y_mask, f_pred_prob, f_pred, cost) = build_model(tparams, model_options, max_word_count)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([xc, mask, y, y_mask], cost, name='f_cost', on_unused_input='warn')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([xc, mask, y, y_mask], grads, name='f_grad', on_unused_input='warn')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        xc, mask, y, y_mask, cost)

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    logging.info("%d train examples" % len(train[0]))
    logging.info("%d valid examples" % len(valid[0]))
    logging.info("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
        logging.info("validFreq auto set to %d", validFreq)
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size
        logging.info("saveFreq auto set to %d", saveFreq)

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.clock()

    trng = RandomStreams(123)

    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                y = [train[2][t] for t in train_index]
                x_c = [train[0][t] for t in train_index]
                x_w = [train[1][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                n_proj = dim_proj_chars# + dim_proj_words
                #def prepare_data(char_seqs, labels, maxw, maxwlen):
                xc, mask, y, y_mask = prepare_data(x_c, y,
                                                   max_word_count,
                                                   max_word_length,
                                                   dim_proj_chars)
                n_samples += xc.shape[1]

                cost = f_grad_shared(xc, mask, y, y_mask)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    logging.error('NaN detected (bad cost)')
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    logging.info('Epoch %d, Update %d, Cost %.4f', eidx, uidx, cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    logging.info('Saving to %s', saveto)

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    logging.info('Incremental save complete')

                if numpy.mod(uidx, validFreq) == 0:
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid, max_word_count, max_word_length, dim_proj_chars)

                    if not pretrain:
                        #train_err = pred_error(f_pred, prepare_data, train, kf, 140, max_word_count, max_word_length, n_proj)
                        test_err = pred_error(f_pred, prepare_data, test, kf_test, max_word_count, max_word_length, dim_proj_chars)
                        history_errs.append([valid_err, test_err])
                    else:
                        history_errs.append([valid_err, 0.0])

                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0
                    if not pretrain:
                        logging.info("Valid %.4f, Test %.4f",
                                     100*(1-valid_err), 100*(1-test_err))
                    else:
                        logging.info("Valid %.4f", 100 * (1-valid_err))

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            logging.warn('Early Stop!')
                            estop = True
                            break

            logging.info('Seen %d samples', n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        logging.warn("Training interrupted")

    end_time = time.clock()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted, max_word_count, max_word_length, dim_proj_chars)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid, max_word_count, max_word_length, dim_proj_chars)
    test_err = pred_error(f_pred, prepare_data, test, kf_test, max_word_count, max_word_length, dim_proj_chars)

    logging.info("Train %.4f, Valid %.4f, Test %.4f",
                                     100*(1-train_err), 100*(1-valid_err), 100*(1-test_err))

    if saveto:
        logging.info("Saving to %s...", saveto)
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    logging.info('The code run for %d epochs, with %f sec/epochs',
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))

    logging.info('Training took %.1fs',
                          (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    a = ArgumentParser("Train/Evaluate the LSTM model")
    a.add_argument("--model", help="Load an existing model")
    a.add_argument("--max-epochs", type=int, default=1000)
    a.add_argument("--pretrain", help="Divide a 90-10 training/eval thing", action="store_true")
    a.add_argument("--words", help="Number of recurrent layers (word level)", type=int, default=0)
    a.add_argument("--letters", help="Number of recurrent layers (letter level)", type=int, default=0)

    p = a.parse_args()

    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=p.max_epochs,
        reload_model=p.model,
        pretrain=p.pretrain,
        word_layers=p.words,
        letter_layers=p.letters
    )
