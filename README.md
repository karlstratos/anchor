Anchor (C++)
=============================
Anchor implements an HMM learning algorithm that makes/exploits an "anchor" assumption: every hidden state has at least 1 observation that can be generated only by that state.
For details, see: Unsupervised Part-Of-Speech Tagging with Anchor Hidden Markov Models (Stratos et al., 2016).

Learning (12 hidden states)
-------------------------
* Learn an HMM using the anchor algorithm:

`./hmm --output /tmp/anchor --data [unlabeled_sequences] --train --unsup anchor --states 12`
* Learn an HMM using the anchor algorithm extended with spelling features (relative weight 0.1):

`./hmm --output /tmp/anchor_extended --data [unlabeled_sequences]  --train --unsup anchor --states 12 --extend cap,hyphen,digit,suff1,suff2,suff3 --extweight 0.1`

* Learn an HMM with Baum-Welch (1000 iterations):

`./hmm --output /tmp/bw --data [unlabeled_sequences] --train --unsup bw --bwiter 1000 --states 12`

* Learn an HMM from clusters:

`./hmm --output /tmp/cluster --data [unlabeled_sequences] --train --unsup cluster --cluster [word_clusters] --states 12`

* Learn a supervised HMM:

`./hmm --output /tmp/supervised --data [labeled_sequences] --train --states 0`

Evaluation
----------

`./hmm --output [trained_model_dir] --data [labeled_sequences] --pred /tmp/pred`

Example
-------

* Learn an HMM with 3 states on sample.txt using the anchor algorithm:

`./hmm --output /tmp/anchor --data sample.txt --train --unsup anchor --states 3`

* Test the model on labeled data sample_answer.txt:

`./hmm --output /tmp/anchor --data sample_answer.txt --pred /tmp/pred`

More Details
------------
Type `./hmm` or `./hmm -h` or `./hmm --help` to see an option menu like this:

    --output [-]:       path to an output directory
    --data [-]:        	path to a data file
    --train:          	train a model?
    --unsup [anchor]:   unsupervised learning method: cluster, bw, anchor
    --states [12]:      number of states (set 0 for supervised training)
    --extend []:      	context extensions (separated by ,)
    --extweight [0.1]:  relative scaling for extended context features
    --cand [300]:   	number of candidates to consider for anchors
    --dev [-]:        	path to a development data file
    --cluster [-]:   	path to clusters (used for unsup=cluster)
    --pred [-]:        	path to a prediction file
    --bwiter [1000]:    maximum number of Baum-Welch iterations
    --triter [1]:     	maximum number of EM iterations for transition estimation
    --help, -h:         show options and quit?
    --detail, -d:    	show detailed options and quit?