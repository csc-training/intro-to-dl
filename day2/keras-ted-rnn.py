# coding: utf-8

# # Ted talks keyword labeling with pre-trained word embeddings
#
# In this script, we'll use pre-trained [GloVe word embeddings]
# (http://nlp.stanford.edu/projects/glove/) for keyword labeling using
# Keras (version $\ge$ 2 is required). This script is largely based on
# the blog post [Using pre-trained word embeddings in a Keras model]
# (https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
# by François Chollet.
#
# **Note that using a GPU with this script is highly recommended.**
#
# First, the needed imports. Keras tells us which backend (Theano,
# Tensorflow, CNTK) it will be using.

from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LSTM, CuDNNLSTM
from keras.utils import to_categorical

from distutils.version import LooseVersion as LV
from keras import __version__
from keras import backend as K

from sklearn import metrics

import xml.etree.ElementTree as ET
import os
import sys

import numpy as np

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

# If we are using TensorFlow as the backend, we can use TensorBoard to
# visualize our progress during training.

if K.backend() == "tensorflow":
    import tensorflow as tf
    from keras.callbacks import TensorBoard
    import os, datetime
    logdir = os.path.join(os.getcwd(), "logs",
                     "ted-cnn-"+datetime.datetime.now()
                          .strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    callbacks = [TensorBoard(log_dir=logdir)]
else:
    callbacks =  None

# ## GloVe word embeddings
#
# Let's begin by loading a datafile containing pre-trained word
# embeddings.  The datafile contains 100-dimensional embeddings for
# 400,000 English words.

GLOVE_DIR = "/wrk/makoskel/glove.6B"

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_dim = len(coefs)
print('Found %d word vectors of dimensionality %d.' % (len(embeddings_index),
                                                       embedding_dim))

# ## Ted talks data set
#
# Next we'll load the TED Talks data set (Kaggle TED Talks, 2017 edition).
# The data is stored in two CSV files, so we load both of them and
# merge them into a single DataFrame.
#xs
# The merged dataset contains transcripts and metadata of 2467 TED talks.
# Each talk is also annotated with a set of tags.

TEXT_DATA_DIR = "/wrk/makoskel/ted/kaggle-ted-talks"

df1 = pd.read_csv(TEXT_DATA_DIR+'/ted_main.csv')
df2 = pd.read_csv(TEXT_DATA_DIR+'/transcripts.csv')
df = pd.merge(left=df1, right=df2, how='inner', left_on='url', right_on='url')

print(len(df), 'talks')

# Now we decide to use either the `transcipt` or the `description` column:

texttype = "transcript"
#texttype = "description"

# ### Keywords
#
# Let's start by converting the string-type lists of tags to Python
# lists.  Then, we take a look at a histogram of number of tags attached
# to talks:

import ast
df['taglist']=df['tags'].apply(lambda x: ast.literal_eval(x))

# We use the `NLABELS` most frequent tags as keyword labels we wish
# to predict:

NLABELS=100

ntags = dict()
for tl in df['taglist']:
    for t in tl:
        if t in ntags:
            ntags[t] += 1
        else:
            ntags[t] = 1

ntagslist_sorted = sorted(ntags, key=ntags.get, reverse=True)
print('Total of', len(ntagslist_sorted), 'tags found. Showing',
      NLABELS, 'most common tags:')
for i, t in enumerate(ntagslist_sorted[:NLABELS]):
    print(i, t, ntags[t])

def tags_to_indices(x):
    ilist = []
    for t in x:
        ilist.append(ntagslist_sorted.index(t))
    return ilist

df['tagidxlist'] = df['taglist'].apply(tags_to_indices)

def indices_to_labels(x):
    labels = np.zeros(NLABELS)
    for i in x:
        if i < NLABELS:
            labels[i] = 1
    return labels

df['labels'] = df['tagidxlist'].apply(indices_to_labels)

# ### Produce input and label tensors
#
# We vectorize the text samples and labels into a 2D integer tensors.
#`MAX_NUM_WORDS` is the number of different words to use as tokens,
# selected based on word frequency. `MAX_SEQUENCE_LENGTH` is the fixed
# sequence length obtained by truncating or padding the original sequences.

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 1000

tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts([x for x in df[texttype]])
sequences = tokenizer.texts_to_sequences([x for x in df[texttype]])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.asarray([x for x in df['labels']])

print('Shape of data tensor:', data.shape)
print('Shape of labels tensor:', labels.shape)

# Next, we split the data into a training set and a validation set.  We
# use a fraction of the data specified by `VALIDATION_SPLIT` for validation.
# Note that we do not use a separate test set in this notebook, due to the
# small size of the dataset.

VALIDATION_SPLIT = 0.2

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
print('Shape of training data tensor:', x_train.shape)
print('Shape of training label tensor:', y_train.shape)
print('Shape of validation data tensor:', x_val.shape)
print('Shape of validation label tensor:', y_val.shape)

# We prepare the embedding matrix by retrieving the corresponding
# word embedding for each token in our vocabulary:

print('Preparing embedding matrix.')

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Shape of embedding matrix:', embedding_matrix.shape)

# ### Initialization

print('Build model...')
model = Sequential()

model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
#model.add(Dropout(0.2))

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(CuDNNLSTM(128))

model.add(Dense(128, activation='relu'))
model.add(Dense(NLABELS, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

print(model.summary())

# ### Learning

epochs = 20
batch_size=16

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    verbose=2, callbacks=callbacks)

# ### Inference
#
# To further analyze the results, we can produce the actual
# predictions for the validation data.

predictions = model.predict(x_val)

# The selected threshold controls the number of label predictions we'll make:

threshold = 0.5

avg_n_gt, avg_n_pred = 0, 0
for t in range(len(y_val)):
    avg_n_gt += len(np.where(y_val[t]>0.5)[0])
    avg_n_pred += len(np.where(predictions[t]>threshold)[0])
avg_n_gt /= len(y_val)
avg_n_pred /= len(y_val)
print('Average number of ground-truth labels per talk: %.2f' % avg_n_gt)
print('Average number of predicted labels per talk: %.2f' % avg_n_pred)

# Let's look at the correct and predicted labels for some talks in the
# validation set.

nb_talks_to_show = 20

for t in range(nb_talks_to_show):
    print(t,':')
    print('    correct: ', end='')
    for idx in np.where(y_val[t]>0.5)[0].tolist():
        sys.stdout.write('['+ntagslist_sorted[idx]+'] ')
    print()
    print('  predicted: ', end='')
    for idx in np.where(predictions[t]>threshold)[0].tolist():
        sys.stdout.write('['+ntagslist_sorted[idx]+'] ')
    print()

# Scikit-learn has some applicable performance [metrics]
# (http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
# we can try:

print('Precision: {0:.3f} (threshold: {1:.2f})'
      .format(metrics.precision_score(y_val.flatten(),
                                      predictions.flatten()>threshold),
              threshold))
print('Recall: {0:.3f} (threshold: {1:.2f})'
      .format(metrics.recall_score(y_val.flatten(),
                                   predictions.flatten()>threshold),
              threshold))
print('F1 score: {0:.3f} (threshold: {1:.2f})'
      .format(metrics.f1_score(y_val.flatten(),
                               predictions.flatten()>threshold),
              threshold))

average_precision = metrics.average_precision_score(y_val.flatten(),
                                                    predictions.flatten())
print('Average precision: {0:.3f}'.format(average_precision))
print('Coverage: {0:.3f}'
      .format(metrics.coverage_error(y_val, predictions)))
print('LRAP: {0:.3f}'
      .format(metrics.label_ranking_average_precision_score(y_val,
                                                            predictions)))
