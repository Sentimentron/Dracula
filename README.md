You're looking at the <code>2016-04-16</code> release of *Dracula*, a part-of-speech tagger optimized for Twitter. This tagger offers very competitive performance whilst only learning character embeddings and neural network weights, meaning it requires considerably less pre-processing that another techniques. This branch represents the release, the actual contents of this branch may change as additional things are documented, but there will be no functional changes.

## Background

Part of speech tagging is a fundamental task in natural language processing, and its part of figuring out the meaning of a particular, for example if the word *heated* represents an adjective ("he was involved in a heated conversation") or a past-tense verb ("the room was heated for several hours"). It's the first step towards a more complete understanding of a phrase through parsing. Tweets are particularly hard to deal with because they contain links, emojis, at-mentions, hashtags, slang, poor capitalisation, typos and bad spelling. 

## How the model works
Unlike most other part of speech taggers, Dracula doesn't look at words directly. Instead, it reads the characters that make up a word and then uses deep neural network techniques to figure out the right tag. [Read more &raquo;](http://dracula.sentimentron.co.uk/how.html)

## Installing the model
You'll need Theano 0.7 or better. [See Theano's installation page for additional details &raquo;](http://deeplearning.net/software/theano/install.html)

## Training the model
Run the <code>train.sh</code> script to train with the default settings. You may need to modify the <code>THEANO_FLAGS</code> variable at the top of this file to suit your hardware configuration (by default, it assumes a single GPU system).

## Assessing the model
1. Start the HTTP server, using <code>THEANO_FLAGS="floatX=float32" python server.py</code>.
2. In another terminal, type <code>python eval.py path/to/assessment/file.conll</code>.
 
### How well does the model perform?
Here's the model's performance for various character embedding sizes. This is assessed using [GATE's TweetIE Evaluation Set](https://gate.ac.uk/wiki/twitie.html) (<code>Data/Gate-Eval.conll</code>).
<table>
<tr><th>Tag</th><th>Size</th><th>Accuracy (% tokens correct)</th><th>Accuracy (% entire sentences correct)</th></tr>
<tr><td><code>2016-04-16-128</code><td>128</td><td>88.69%</td><td>20.33%</td></tr>
<tr><td><code>2016-04-16-64</code><td>64</td><td>87.29%</td><td>16.10%</td></tr>
<tr><td><code>2016-04-16-32</code><td>32</td><td>84.98%</td><td>11.86%</td></tr>
<tr><td><code>2016-04-16-16</code><td>16</td><td>74.24%</td><td>3.39%</td></tr>
</table>


## Changing the the embedding size
Make the following modifications:
* <code>server.py</code>, in the <code>prepare_data</code> call on line 122, change <code>32</code> (the last argument) to the correct size.
* <code>lstm.py</code>, in the <code>train_lstm</code> arguments on line 104, change <code>dim_proj_chars</code> default value to the correct size.

## Licensing
All the code in this repository is distributed under the terms of `LICENSE.md`.

## Acknowledgements, references
The code in <code>lstm.py</code> is a heavily modified version of [Pierre Luc Carrier and Kyunghyun Cho's _LSTM Networks for Sentiment Analysis_ tutorial](http://deeplearning.net/tutorial/lstm.html). 

* [Hochreiter, S., &amp; Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
* [Gers, F. A., Schmidhuber, J., &amp; Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. Neural computation, 12(10), 2451-2471.](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
* [Bastien, Frédéric, Lamblin, Pascal, Pascanu, Razvan, Bergstra, James, Goodfellow, Ian, Bergeron, Arnaud, Bouchard, Nicolas, and Bengio, Yoshua. Theano: new features and speed improvements. NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2012.](http://www.iro.umontreal.ca/~lisa/pointeurs/nips2012_deep_workshop_theano_final.pdf)
* [Bergstra, James, Breuleux, Olivier, Bastien, Frédéric, Lamblin, Pascal, Pascanu, Razvan, Desjardins, Guillaume, Turian, Joseph, Warde-Farley, David, and Bengio, Yoshua. Theano: a CPU and GPU math expression compiler. In Proceedings of the Python for Scientific Computing Conference (SciPy), June 2010.](http://www.iro.umontreal.ca/~lisa/pointeurs/theano_scipy2010.pdf)

The inspiration for using character embeddings to do this job is from C. Santos' series of papers linked below. 

* [C. Santos and B. Zadrozny, "Learning character-level representations for part-of-speech tagging", Proceedings of the 31st International Conference on Machine Learning (ICML-14), pp. 1818--1826, 2014.](http://machinelearning.wustl.edu/mlpapers/papers/icml2014c2_santos14)
* [C. D. Santos and M. Gatti, "Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts", COLING, pp. 69-78, 2014.](http://www.anthology.aclweb.org/C/C14/C14-1008.pdf)

Finally, GATE gathered the the most important corpora used for training, and provide a reference benchmark:
* [L. Derczynski, A. Ritter, S. Clark and K. Bontcheva, "Twitter Part-of-Speech Tagging for All: Overcoming Sparse and Noisy Data.", RANLP, pp. 198--206, 2013.](https://gate.ac.uk/sale/ranlp2013/twitter_pos/twitter_pos.pdf)

