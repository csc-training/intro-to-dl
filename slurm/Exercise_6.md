# Exercise 6

In this exercise, we study text categorization using the [20
newsgroups](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)
dataset.

## Files

### TF2/Keras

* *tf2-20ng-rnn.py*: 20 newsgroups classification with a RNN
* *tf2-20ng-cnn.py*: 20 newsgroups classification with a CNN
* *tf2-20ng-bert.py*: 20 newsgroups classification with BERT finetuning

### PyTorch

* *pytorch_20ng_rnn.py*: 20 newsgroups classification with a RNN
* *pytorch_20ng_cnn.py*: 20 newsgroups classification with a CNN
* *pytorch_20ng_bert.py*: 20 newsgroups classification with BERT finetuning

## Tasks

### Task 1

Run all three models and compare their accuracies and runtimes.

### Task 2

Pick one model and try to improve the results, e.g., by tweaking the model or the training parameters (optimizer, batch size, number of epochs, etc.).
You can also work on replacing BERT with another Transformers model; see [documentation](https://huggingface.co/transformers/).
