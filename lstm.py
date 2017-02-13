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
from theano.tensor.nnet import sigmoid

from util import get_minibatches_idx
from modelio import *

from nn_layers import *
from nn_lstm import lstm_layer, lstm_unmasked_layer, bidirectional_lstm_layer
from nn_params import *
from nn_optimizers import *
from nn_support import pred_error
from nn_serialization import zipp, unzip, load_params

import os.path
import pickle
import random

from theano.compile.nanguardmode import NanGuardMode

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def build_model(tparams, options, maxw, training=True):
    xc0 = tensor.tensor3('xc0', dtype='int8')
    xc1 = tensor.tensor3('xc1', dtype='int8')
    mask0 = tensor.tensor4('mask0', dtype=config.floatX)
    mask1 = tensor.tensor4('mask1', dtype=config.floatX)
    y = tensor.vector('y', dtype='int8')
    y_mask = tensor.vector('y_mask', dtype='float32')

    n_batch = xc0.shape[2]

    emb0 = embeddings_layer(xc0, tparams['Cemb'], options['dim_proj'])
    emb1 = embeddings_layer(xc1, tparams['Cemb'], options['dim_proj'])

    dist0 = emb0
    dist_mask0 = mask0
    dist0 = dist0 * dist_mask0
    dist1 = emb1
    dist_mask1 = mask1
    dist1 = dist1 * dist_mask1

    dist0 = dist0.dimshuffle(0, 2, 1, 3)
    dist1 = dist1.dimshuffle(0, 2, 1, 3)

    def _conv2d_step(x_):
        p_ = tparams['conv']
        x_ = x_.dimshuffle(0, 'x', 1, 2)
        return tensor.nnet.conv2d(x_, p_, border_mode='valid')

    dist0, updates0 = theano.scan(_conv2d_step, sequences=[dist0], n_steps=dist0.shape[0])
    dist1, updates1 = theano.scan(_conv2d_step, sequences=[dist1], n_steps=dist0.shape[0])

    dist0 = dist0.flatten(3)
    dist1 = dist1.flatten(3)

    proj20 = dist0
    proj21 = dist1

    for i in range(options['word_layers']):
        name = 'lstm_words_%d' % (i + 1,)
        proj20 = bidirectional_lstm_layer(tparams, proj20, options, name)
        proj21 = bidirectional_lstm_layer(tparams, proj21, options, name)

    proj20 = proj20.mean(axis=0, keepdims=True)
    proj21 = proj21.mean(axis=0, keepdims=True)
#   tmp = proj20-proj21
    #tmp = tensor.concatenate([proj20, proj21], axis=2)
#    tmp = proj20-proj21
#   pred = softmax_layer(tmp, tparams['U'], tparams['b'], y_mask, maxw, training)
#    pred = theano.printing.Print("pred", attrs=["shape"])(pred)

    def _sqr_mag(x):
        return tensor.sqr(x).sum(axis=-1)

    def _mag(x):
        return tensor.sqrt(tensor.maximum(_sqr_mag(x), numpy.finfo(x.dtype).tiny))

    def _cosine(x, y):
        return tensor.clip((1 - (x * y).sum(axis=-1) / (_mag(x) * _mag(y)))/ 2, 0, 1)

    pred30 = tensor.dot(proj20, tparams['U']) + tparams['b']
    pred31 = tensor.dot(proj21, tparams['U']) + tparams['b']
#    pred30 = softmax_layer(proj20, tparams['U'], tparams['b'], y_mask, maxw, training)
#    pred31 = softmax_layer(proj21, tparams['U'], tparams['b'], y_mask, maxw, training)
    #pred = _cosine(pred30, pred31)
    pred = _mag(pred30 - pred31)
    dpred = pred > 0.5
    #pred = tensor.nnet.sigmoid(pred)

    f_pred_prob = theano.function([xc0, xc1, mask0, mask1, y_mask], pred, name='f_pred_prob', on_unused_input='ignore')
    f_pred = theano.function([xc0, xc1, mask0, mask1, y_mask], dpred, name='f_pred', on_unused_input='ignore')

    #cost = tensor.nnet.binary_crossentropy(y, 1 - tensor.nnet.sigmoid(pred)).mean()
    #cost = tensor.sqr(pred * y * 0.9 + pred * (1 - y) * 0.1).mean()
    #cost = (tensor.sqr(dpred - y) * pred).mean()
    cost = (tensor.sqr(y - pred)).mean()

    return xc0, xc1, mask0, mask1, y, y_mask, f_pred_prob, f_pred, cost

def split_at(src, prop):
    # Everything's preshuffled in this model
    valid_indices = [i for i, _ in enumerate(zip(src[0], src[1]))]
    src_chars0, src_chars1, src_labels = [], [], []
    val_chars0, val_chars1, val_labels = [], [], []
    fin = max(int(prop * len(src[0])), 1)
    print len(src[0]), prop, fin
    for i, idx in enumerate(valid_indices):
        c0 = src[0][idx]
        c1 = src[1][idx]
        l = src[2][idx]
        if i < fin:
            val_chars0.append(c0)
            val_chars1.append(c1)
            val_labels.append(l)
        else:
            src_chars0.append(c0)
            src_chars1.append(c1)
            src_labels.append(l)
    return (src_chars0, src_chars1, src_labels), (val_chars0, val_chars1, val_labels)

def train_lstm(
    dim_proj_chars=8,  # character embedding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=8,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.0001,  # Weight decay for the classifier applied to the U weights.
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=900,  # Compute the validation error after this number of update.
    saveFreq=2220,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=8,  # The batch size during training.
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
    pos_dict = {},
    word_layers=0,
    letter_layers=0,
    evaluating=False # If true, compute train, test and valid accuracy only
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    if reload_model is not None:
        load_params(reload_model, model_options)
        char_dict = model_options['char_dict']
        pos_dict = model_options['pos_dict']

    # Load the training data
    print 'Loading data'

    input_path = "Data/quora_duplicate_questions_stripped.tsv"

    load_data(input_path, char_dict)

    max_word_count = 0
    max_word_length = 0
    max_length = 0
    # Now load the data for real
    data = load_data(input_path, char_dict)
    train, eval = split_at(data, 0.05)
    test, valid = split_at(eval, 0.50)
    max_word_count = max(max_word_count, \
    get_max_word_count(input_path))
    max_word_length = max(max_word_length, \
    get_max_word_length(input_path))
    # This parameter is for information only
    max_length = max(max_word_length, get_max_length(input_path))

    #print numpy.max(train[2])
    ydim = 128
    print "ydim =", ydim

    model_options['ydim'] = ydim
    model_options['n_chars'] = len(char_dict)+1
    model_options['max_letters'] = max_word_length

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
    (xc0, xc1, mask0, mask1,
     y, y_mask, f_pred_prob, f_pred, cost) = build_model(tparams, model_options, max_word_count, training=not evaluating)
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    logging.info("%d train examples" % len(train[0]))
    logging.info("%d valid examples" % len(valid[0]))
    logging.info("%d test examples" % len(test[0]))

    if evaluating:
        input_path = "Data/quora_duplicate_questions_eval.tsv"
        eval_data = load_data(input_path, char_dict)
        kf_eval = get_minibatches_idx(len(eval_data[0]), valid_batch_size)

        kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
        valid_err = pred_error(f_pred, prepare_data, valid, kf_valid, max_word_count, max_word_length, dim_proj_chars)
        test_err = pred_error(f_pred, prepare_data, test, kf_test, max_word_count, max_word_length, dim_proj_chars)

        logging.info("Valid %.4f, Test %.4f",
                     100*(1-valid_err), 100*(1-test_err))

        guess_err = pred_error(f_pred, prepare_data, eval_data, kf_eval, max_word_count, max_word_length, dim_proj_chars)
        logging.info("Guesstimate error: %.4f",
                     100*(1-guess_err))
        return

    if decay_c > 0:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0
        weight_decay += (tparams['U']**2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([xc0, xc1, mask0, mask1, y, y_mask], cost, name='f_cost', on_unused_input='warn', mode=NanGuardMode(nan_is_error=True))

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([xc0, xc1, mask0, mask1, y, y_mask], grads, name='f_grad', on_unused_input='warn')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        xc0, xc1, mask0, mask1, y, y_mask, cost)


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
                x_c_0 = [train[0][t] for t in train_index]
                x_c_1 = [train[1][t] for t in train_index]
                y = [train[2][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                n_proj = dim_proj_chars# + dim_proj_words
                #def prepare_data(char_seqs, labels, maxw, maxwlen):
                xc0, mask0, y_,  y_mask_ = prepare_data(x_c_0, y,
                                                   max_word_count,
                                                   max_word_length,
                                                   dim_proj_chars)
                xc1, mask1, _, _ = prepare_data(x_c_1, y,
                                                max_word_count,
                                                max_word_length,
                                                dim_proj_chars)
                y, y_mask = y_, y_mask_
                n_samples += xc0.shape[1]

                cost = f_grad_shared(xc0, xc1, mask0, mask1, y, y_mask)
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
    a.add_argument("--evaluate", default=False, action='store_true')

    p = a.parse_args()

    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=p.max_epochs,
        reload_model=p.model,
        pretrain=p.pretrain,
        word_layers=p.words,
        letter_layers=p.letters,
        evaluating=p.evaluate
    )
