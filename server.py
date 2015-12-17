__author__ = 'rtownsend'

from lstm import build_model
from modelio import string_to_unprepared_format, prepare_data
from nn_params import *
from nn_optimizers import *
from nn_serialization import load_params
from matcher import MultiSimilarityMatcher

from flask import Flask, request, jsonify
app = Flask(__name__)

from collections import defaultdict, Counter

model = None

def get_lstm(
    dim_proj_chars=16,  # character embedding dimension and LSTM number of hidden units.
    dim_proj_words=16,
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.0001,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    model_path='lstm_model.npz', # Where to load the model from
    batch_size=20,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    # Use to keep track of feature enumeration
    char_dict = {},
    word_dict = {},
    pos_dict = {},
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    load_params(model_path, model_options)
    char_dict = model_options['char_dict']
    word_dict = model_options['word_dict']
    for w in word_dict.keys():
        word_dict[w.decode('utf8')] = word_dict[w]
    pos_dict = model_options['pos_dict']

    print "Creating matcher..."
    matcher = MultiSimilarityMatcher()
    matcher.update_from_dict(word_dict)
    model_options['matcher'] = matcher

    print "Continuing with model..."
    ydim = 27 # Hard-code, one that appears in the testing set, not in the training set

    model_options['ydim'] = ydim
    model_options['n_chars'] = len(char_dict)+1
    model_options['n_words'] = len(word_dict)+1

    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options, True)

    logging.info('Building model')

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (dropout_mask, xc, xw, mask, wmask,
     y, y_mask, f_pred_prob, f_pred, cost) = build_model(tparams, model_options, 38)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    inv_tag_dict = {}
    for c in pos_dict:
        inv_tag_dict[pos_dict[c]] = c

    model_options['inv_pos_dict'] = inv_tag_dict

    return dropout_mask, xc, xw, mask, wmask, y, y_mask, f_pred_prob, f_pred, cost, model_options

@app.route("/api/tag", methods=["GET"])
def hello():
    global model

    text = request.args.get("text", "")

    response = {}
    if text is None or len(text) == 0:
        response["error"] = "no text provided"
    response["text"] = text

    text = text.split()
    rebuilt_text = []
    for t in text:
        if t not in model[-1]['word_dict']:
            matcher = model[-1]['matcher']
            _, r = matcher.get_most_similar_word(t)
            rebuilt_text.append(r.decode('utf8'))
        else:
            rebuilt_text.append(t)

    rebuilt_text = ' '.join(rebuilt_text)
    response['prepared_text'] = rebuilt_text
    errors, chars, words, labels = string_to_unprepared_format(rebuilt_text, model[-1]['char_dict'], model[-1]['word_dict'])
    if len(errors) > 0:
        response["tokenization_errors"] = errors

    print chars, words
    xc, xw, mask, wmask, y, y_mask = prepare_data(chars, words, labels, 140, 38, 32)

    print model[-1]['Cemb']

    dropout_mask = numpy.ones(model[-1]['U'].shape).astype(dtype=theano.config.floatX) * 0.5

    pred = model[-3](dropout_mask, xc, xw, mask, wmask, y_mask)
    print pred

    words, windows = pred.shape
    print windows, words
    tag_counter = defaultdict(Counter)

    # Scan along each 16-word window,
    # build up a list of the most popular tags
    for winidx in range(windows):
        for idx, i in enumerate(pred[:, winidx]):
            wordidx = winidx + idx
            if i == 0:
                break
            t = model[-1]['inv_pos_dict'][i]
            tag_counter[wordidx].update([t])

    # Set up the response
    response['tags'] = []
    response['tags_and_text'] = []

    text = text
    for idx in sorted(tag_counter):
        if len(tag_counter[idx]) == 0:
            break
        tag, _ = tag_counter[idx].most_common()[0]
        response['tags'].append(tag)
        response['tags_and_text'].append((tag, text[idx]))

    if False:
        for idx, i in enumerate(pred[:, 0]):
            if i == 0:
                break
            t = model[-1]['inv_pos_dict'][i]
            response['tags'].append(t)
            response['tags_and_text'].append((t, text[idx]))
    return jsonify(**response)

if __name__ == "__main__":
    global model

    logging.basicConfig(level=logging.DEBUG)

    # Build the model
    model = get_lstm(reload_model="pretrained_model.npz")

    print "Starting server..."
    app.run(debug=True)
