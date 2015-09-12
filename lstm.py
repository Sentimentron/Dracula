'''
Build a tweet sentiment analyzer
'''

from argparse import ArgumentParser
import logging

import cPickle as pkl
import time

import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from util import get_minibatches_idx, numpy_floatX
from modelio import load_pos_tagged_data, prepare_data

from nn_layers import *
from nn_lstm import lstm_layer
from nn_params import *
from nn_optimizers import *
from nn_support import pred_error
from nn_serialization import zipp, unzip, load_params

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)



def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int8')
    x.tag.test_value=numpy.asarray([[5, 5, 1], [5, 5, 1], [5, 5, 1]])
    mask = tensor.matrix('mask', dtype=config.floatX)
    mask.tag.test_value=numpy.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    wmask = tensor.matrix('wmask', dtype='int8')
    wmask.tag.test_value = numpy.random.randint(0, 13, (6, 4))
    y = tensor.matrix('y', dtype='int8')
    y.tag.test_value=numpy.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    y_mask = tensor.matrix('y_mask', dtype='int8')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    dim = options['dim_proj_chars']

    emb = embeddings_layer(x, tparams['Cemb'], n_timesteps, n_samples, options['dim_proj_chars'])

    proj = lstm_layer(tparams, emb, options, "lstm", mask=mask)

    proj = lstm_mask_layer(proj, mask)

    avg_per_word = per_word_averaging_layer(proj, wmask, n_samples, dim)

    pred = softmax_layer(avg_per_word, tparams['U'], tparams['b'], y_mask)

    f_pred_prob = theano.function([x, mask, wmask, y_mask], pred, name='f_pred_prob', on_unused_input='ignore')
    f_pred = theano.function([x, mask, wmask, y_mask], pred.argmax(axis=2), name='f_pred', on_unused_input='ignore')

    def cost_scan_i(i, j, free_var):
        return -tensor.log(i[tensor.arange(n_samples), j] + 1e-8)

    cost, _ = theano.scan(cost_scan_i, outputs_info=None, sequences=[pred, y, tensor.arange(n_samples)])

    cost = cost.mean()

    return use_noise, x, mask, wmask, y, y_mask, f_pred_prob, f_pred, cost

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
    dim_proj_chars=12,  # character embedding dimension and LSTM number of hidden units.
    dim_proj_words=128,
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.0001,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
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
    test = load_pos_tagged_data("Data/TweeboDaily547.conll", char_dict, pos_dict)

    ydim = numpy.max(numpy.amax(train[1])) + 1
    ydim = 26 # Hard-code, one that appears in the testing set, not in the training set

    model_options['ydim'] = ydim
    model_options['n_chars'] = len(char_dict)+1

    logging.info('Building model')
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

                cost = f_grad_shared(x, mask, wmask, y, y_mask)
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

                    logging.info("Train %.4f, Valid %.4f, Test %.4f",
                                 100*(1-train_err), 100*(1-valid_err), 100*(1-test_err))

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

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

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
    a.add_argument("--max-epochs", type=int, default=100)

    p = a.parse_args()

    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=p.max_epochs,
        reload_model=p.model
    )
