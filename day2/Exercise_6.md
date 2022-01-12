# Exercise 6

In this exercise, we study text categorization using the [_20
newsgroups_](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)
(20ng) dataset. The dataset contains 20,000 text documents (Usenet messages)
in 20 categories (newsgroups or topics). 

## Task 1

Try three different approaches for text classification with the _20 newsgroups_
(20ng) dataset:

- Recurrent neural network (RNN): [tf2-20ng-rnn.py](tf2-20ng-rnn.py)
- Convolutional neural network (CNN): [tf2-20ng-cnn.py](tf2-20ng-cnn.py)
- BERT finetuning: [tf2-20ng-bert.py](tf2-20ng-bert.py)

Run all three models and compare their accuracies and run times. (There is no
separate evaluation script this time, test set accuracies are printed at the end
of the run.)

<details><summary><b>How to do the same with PyTorch</b></summary>

Corresponding PyTorch scripts:

- Recurrent neural network (RNN): [pytorch_20ng_rnn.py](pytorch_20ng_rnn.py)
- Convolutional neural network (CNN): [pytorch_20ng_cnn.py](pytorch_20ng_cnn.py)
- BERT finetuning: [pytorch_20ng_bert.py](pytorch_20ng_bert.py)

</details>


## Task 2

Pick one model (RNN, CNN or BERT) and try to improve the results, e.g., by
tweaking the model or the training parameters (optimizer, batch size, number of
epochs, etc.). 

You can also work on replacing BERT with another Transformers model (for example
[DistilBert](https://huggingface.co/docs/transformers/master/en/model_doc/distilbert)). 
See also the [HuggingFace Transformers documentation](https://huggingface.co/transformers/).

## Extracurricular

See <https://keras.io/examples/nlp/> for more Keras examples on natural language
processing.
