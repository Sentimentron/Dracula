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
from nn_lstm import lstm_layer, lstm_unmasked_layer, bidirectional_lstm_layer, lstm_bidirectional_layer
from nn_params import *
from nn_optimizers import *
from nn_support import pred_error
from nn_serialization import zipp, unzip, load_params

from tensorflow.models.rnn import rnn_cell

import os.path
import pickle

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def build_model(tparams, options, maxw, training=True):

    xc = tf.placeholder('int32', shape=[options['max_word_idx'], None, None])
    mask = tf.placeholder('float32', shape=[options['max_word_idx'], options['max_char_idx'], None, options['dim_proj']])
    y = tf.placeholder('int32', shape=[options['max_word_idx'], None], name='y')
    y_mask = tf.placeholder('float32', shape=[options['max_word_idx'], None], name='y_mask')

    #tparams['Cemb'] = tf.Print(tparams['Cemb'], [tparams['Cemb']], message="Cemb")
    emb = embeddings_layer(xc, tparams['Cemb'])

    dist = emb
    dist_mask = mask

    #dist = tf.mul(dist, dist_mask)

    if options['letter_layers'] > 0:
        name = "lstm_chars_1"
        dist = lstm_bidirectional_layer(tparams, dist, options, prefix=name)

    dist = per_word_averaging_layer(dist, dist_mask)

    proj2 = dist

    if False:
        for i in range(options['word_layers']):
            name = 'lstm_words_%d' % (i + 1,)
            proj2 = bidirectional_lstm_layer(tparams, proj2, options, name)

            #    tparams['U'] = tf.Print(tparams['U'], [tparams['U']], message="U=")
    pred_prob = softmax_layer(proj2, tparams['U'], tparams['b'], y_mask, maxw, True, False)
    pred_logits = softmax_layer(proj2, tparams['U'], tparams['b'], y_mask, maxw, True, True)

    def cost_fn(cur):
        # TensorFlow does not support negative indexing WTF
        # labels = cur[:, :, -1]
        labels = cur[:, 46] # TODO: don't hard-code me
        labels = tf.cast(labels, dtype='int32')
        preds = cur[:, :46]
        # The non-sparse version of this: y
        return tf.nn.sparse_softmax_cross_entropy_with_logits(preds, labels)

    # Because you can't iterate over two tensors together, concat them
    y = tf.cast(y, dtype='float32', name="y_casted_to_float32")
    y_expanded = tf.expand_dims(y, -1, name="y_expanded")
    #y = tf.expand_dims(y, 2, name="y_expanded_dims")
    packed_format = tf.concat(2, [pred_logits, y_expanded], name="y_concatenated_with_logits")
    #packed_format = tf.Print(packed_format, [tf.shape(packed_format)], message="packed_format=")
    #    cost = tf.map_fn(cost_fn, packed_format)
    #costs = [cost_fn(packed_format[i, :, :]) for i in range(50)] # TODO: don't hard-code me
    costs = tf.map_fn(cost_fn, packed_format)
    #cost = tf.reduce_mean(tf.concat(0, costs))
    cost = tf.reduce_mean(costs)

    #    cost_logits = -tf.log(tf.transpose(pred_prob, [2, 0, 1]) + 1e-8)
    #cost_logits = tf.Print(cost_logits, [tf.shape(cost_logits)], message="cost_logits = ")
    #y = tf.Print(y, [y], message="y = ")
    #y_eq_zero = tf.eq(y, 0)
    #y_eq_zero_idx = tf.where(y_eq_zero)
    #costs = tf.gather(cost_logits, y)
    #    costs = tf.mul(costs, y_mask)
    #costs = tf.Print(costs, [costs], message="costs")
    #cost = tf.reduce_mean(costs)
    pred = tf.argmax(pred_prob, 2)

    #   cost = -tf.reduce_mean(tf.mul(y_mask, pred_prob[y]))

    #def cost_fn(i):
    #    return tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logits[i],\
    #    y[:, i])

    #cost = tf.map_fn(cost_fn, tf.range(0, 50))

#    cost = tf.zeros_like(pred, dtype='float32')
#    for i in range(50):
#        cost = tf.add(cost, tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logits[:,
#        i, :], y[:, i]))

    #cost = tf.reduce_sum(tf.div(cost, 50))
#    cost = tf.map_fn(lambda x: cost_fn(x, pred_logits, y), tf.range(0, 4))
#    cost = tf.reduce_mean(cost)

    return xc, mask, y, y_mask, pred_prob, pred, cost

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
    valid_batch_size=50,  # The batch size used for validation/test set.
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

    input_path = "Data/Gate.conll"

    load_pos_tagged_data(input_path, char_dict, word_dict, pos_dict)

    with open("substitutions.pkl", "rb") as fin:
        word_dict = pickle.load(fin)

    max_word_count = 32
    max_word_length = 20
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

    max_word_count = 8
    max_word_length = 8

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

    model_options['max_word_idx'] = max_word_count
    model_options['max_char_idx'] = max_word_length
    model_options['batch_size'] = batch_size


    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options, reload_model is not None)

    print params.keys()

    logging.info("Configuring minibatches...")
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

    logging.info('Building model')
    with tf.Graph().as_default():

        tparams = init_tparams(params)
        # Initialize the TensorFlow session
        sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
        # Create TensorFlow variables from the parameters
        saver = tf.train.Saver(tparams)
        (xc, mask,
        y, y_mask, f_pred_prob, f_pred, cost) = build_model(tparams, model_options, max_word_count)
        opt = tf.train.AdamOptimizer()
        train_step = opt.minimize(cost)
        sess = sm.prepare_session("", init_op=tf.initialize_all_variables(),
                                      saver=saver)
        def pred_error(f_pred, prepare_data, data, iterator, maxw, \
        max_word_length, dim_proj):
            valid_err = []
            valid_shapes = []
            for _, valid_index in iterator:
                _xc, _mask, _y, _y_mask = prepare_data([data[0][t] for t in \
                valid_index], numpy.array(data[2])[valid_index], maxw, \
                max_word_length, dim_proj)
                cur_feed_dict = {
                    xc: _xc, mask: _mask, y: _y, y_mask: _y_mask
                }
                preds = sess.run(f_pred, \
                feed_dict=cur_feed_dict)
                #                for i, j in zip(preds.flatten(), _y.flatten()):
                #    print i, j, i == j, (i == 0) and (j == 0)
                preds = preds[numpy.nonzero(_y_mask)]
                assert numpy.max(preds) < 47
                assert numpy.max(_y) < 47
                #                print preds, _y[numpy.nonzero(_y_mask)]
                acc = numpy.equal(preds, _y[numpy.nonzero(_y_mask)])
                valid_shapes.append(preds.size)
                valid_err.append(acc.sum())
            valid_err = 1. - 1.0*numpy.asarray(valid_err).sum() /\
            numpy.asarray(valid_shapes).sum()
            return valid_err


        if decay_c > 0.:
            decay_c = tf.constant(numpy_floatX(decay_c), name='decay_c', \
            dtype='float32')
            weight_decay = 0.
            weight_decay += tf.reduce_sum(tparams['U'] ** 2)
            weight_decay *= decay_c
            cost += weight_decay

        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                _y = [train[2][t] for t in train_index]
                x_c = [train[0][t] for t in train_index]
                x_w = [train[1][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                n_proj = dim_proj_chars# + dim_proj_words
                #def prepare_data(char_seqs, labels, maxw, maxwlen):
                _xc, _mask, _y, _y_mask = prepare_data(x_c, _y,
                                                   max_word_count,
                                                   max_word_length,
                                                   dim_proj_chars)

                cur_feed_dict = {
                    xc: _xc, mask: _mask, y: _y, y_mask: _y_mask
                }

                sess.run(train_step, feed_dict=cur_feed_dict)
                """                train_step.run(cur_feed_dict, session=sess)
                grads_and_vars = opt.compute_gradients(cost, tf.trainable_variables())
                grads = opt.apply_gradients(grads_and_vars)
                sess.run([grads], feed_dict=cur_feed_dict)"""

                if numpy.mod(uidx, dispFreq) == 0:
#                    _cost = cost(xc=_xc, mask=_mask, y=_y, y_mask=_y_mask)
                    _cost = sess.run([cost], feed_dict=cur_feed_dict)[0]
                    logging.info('Epoch %d, Update %d, Cost %.4f', eidx, uidx,
                    _cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    logging.info('Saving to %s', saveto)

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    saver.save(sess, saveto)
                    logging.info('Incremental save complete')

                if numpy.mod(uidx, validFreq) == 0:
                    valid_err = pred_error(f_pred, prepare_data, valid,
                    kf_valid, max_word_count, max_word_length, dim_proj_chars)

                    if not pretrain:
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
                    saver.save(sess, saveto)
                    #umpy.savez(saveto, train_err=train_err,
                    #           valid_err=valid_err, test_err=test_err,
                    #           history_errs=history_errs, **best_p)
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
