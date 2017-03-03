You're looking at <code>quora-questions</code>, a special version of *Dracula* which is designed to assess whether two questions are the same or not. It does this by building word-level, then document-level representations.

[More information about how it works &raquo;](https://medium.com/@sentimentron/tackling-the-quora-questions-dataset-43666c74bb0e)

## Available versions
* <code>quora-model-1</code> implements a softmax decision and 64-size character embeddings. [Download the model &raquo;](http://dracula.sentimentron.co.uk/quora-models/quora-model-1.npz) (Approx. 350 MB)
* *Recommended*: <code>quora-model-2</code> implements a Euclidean distance and 8-size character embeddings. [Download the model &raquo;](http://dracula.sentimentron.co.uk/quora-models/quora-model-2-8.npz) (Approx. 2 MB)
* <code>quora-model-2-64</code> is the same, but with 64-size embeddings. [Download &raquo;](http://dracula.sentimentron.co.uk/quora-models/quora-model-2-64.npz) (Approx 350 MB)

## Evaluating
You'll need Theano 0.7 or better. [See Theano's installation page for additional details &raquo;](http://deeplearning.net/software/theano/install.html)

    python lstm.py --words 1 --model lstm_model.npz --evaluate

