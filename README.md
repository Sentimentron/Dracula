You're looking at the <code>2016-04-16</code> release of *Dracula*, a part-of-speech tagger optimized for Twitter. This branch represents the release, the actual contents of this branch may change as additional things are documented, but there will be no functional changes.

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
