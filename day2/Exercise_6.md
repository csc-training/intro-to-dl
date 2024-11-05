# Exercise 6

In this exercise, we study text categorization using the [_20
newsgroups_](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)
(20ng) dataset. The dataset contains 20,000 text documents (Usenet messages)
in 20 categories (newsgroups or topics). For the embeddings of RNNs and CNNs we are using pre-trained 100-dimensional [GloVe](https://nlp.stanford.edu/projects/glove/) vectors.

## Task 1

Try three different approaches for text classification with the _20 newsgroups_
(20ng) dataset:

- Recurrent neural network (RNN): [pytorch_20ng_rnn.py](pytorch_20ng_rnn.py)
- BERT finetuning: [pytorch_20ng_bert.py](pytorch_20ng_bert.py)
- Convolutional neural network (CNN): [pytorch_20ng_cnn.py](pytorch_20ng_cnn.py)

Run all three models and compare their accuracies and run times.

## Task 2

Pick one model (RNN, CNN or BERT) and try to improve the results, e.g., by
tweaking the model or the training parameters (optimizer, batch size, number of
epochs, etc.). 

You can also work on replacing BERT with another Transformers model (for example
[DistilBert](https://huggingface.co/docs/transformers/master/en/model_doc/distilbert)). 
See also the [HuggingFace Transformers documentation](https://huggingface.co/transformers/).

