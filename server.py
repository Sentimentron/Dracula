__author__ = 'rtownsend'

from lstm import build_model
from modelio import string_to_unprepared_format, prepare_data
from nn_params import *
from nn_optimizers import *
from nn_serialization import load_params

from flask import Flask, request, jsonify
app = Flask(__name__)

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
    saveto='lstm_model.npz',  # The best model will be saved there
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

    load_params('pretrained_model.npz', model_options)
    char_dict = model_options['char_dict']
    word_dict = model_options['word_dict']
    pos_dict = model_options['pos_dict']

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
     y, y_mask, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

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

@app.route("/api/tag/<text>")
def hello(text):
    global model

    response = {}
    if text is None:
        text = request.args.get("text", "")
    if text is None or len(text) == 0:
        response["error"] = "no text provided"
    elif len(text) > 140:
        response["error"] = "text is too long (truncated to 140)"
        text = ''.join(text[:140])
    response["text"] = text

    errors, chars, words, labels = string_to_unprepared_format(text, model[-1]['char_dict'], model[-1]['word_dict'])
    if len(errors) > 0:
        response["tokenization_errors"] = errors

    print chars, words
    xc, xw, mask, wmask, y, y_mask = prepare_data(chars, words, labels)

    print model[-1]['Cemb']

    dropout_mask = numpy.ones(model[-1]['U'].shape).astype(dtype=theano.config.floatX) * 0.5

    pred = model[-3](dropout_mask, xc, xw, mask, wmask, y_mask)
    print pred

    response['tags'] = []
    for i in pred[:, 0]:
        if i == 0:
            break
        t = model[-1]['inv_pos_dict'][i]
        response['tags'].append(t)
    return jsonify(**response)

if __name__ == "__main__":
    global model

    logging.basicConfig(level=logging.DEBUG)

    # Build the model
    model = get_lstm(reload_model="pretrained_model.npz")

    print "Starting server..."
    app.run(debug=True)
