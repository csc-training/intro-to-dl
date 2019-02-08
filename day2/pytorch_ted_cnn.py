# coding: utf-8

# # TED Talks keyword labeling with pre-trained word embeddings
# 
# In this notebook, we'll use pre-trained [GloVe word
# embeddings](http://nlp.stanford.edu/projects/glove/) for keyword
# labeling using PyTorch. This notebook is largely based on the blog
# post [Using pre-trained word embeddings in a Keras model]
# (https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
# by François Chollet.
# 
# **Note that using a GPU with this notebook is highly recommended.**
# 
# First, the needed imports.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from distutils.version import LooseVersion as LV

from keras.preprocessing import sequence, text

from sklearn import metrics

import os
import sys

import pandas as pd
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert(LV(torch.__version__) >= LV("1.0.0"))

# TensorBoard is a tool for visualizing progress during training.
# Although TensorBoard was created for TensorFlow, it can also be used
# with PyTorch.  It is easiest to use it with the tensorboardX module.

try:
    import tensorboardX
    import os, datetime
    logdir = os.path.join(os.getcwd(), "logs",
                          "ted-cnn-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    log = tensorboardX.SummaryWriter(logdir)
except ImportError as e:
    log = None

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

# ## TED Talks data set
# 
# Next we'll load the TED Talks data set (Kaggle [TED Talks]
# (https://www.kaggle.com/rounakbanik/ted-talks), 2017 edition).
# The data is stored in two CSV files, so we load both of them and
# merge them into a single DataFrame.
# 
# The merged dataset contains transcripts and metadata of 2467 TED
# talks. Each talk is also annotated with a set of tags.

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
# lists.  Then, we take a look at a histogram of number of tags
# attached to talks:

import ast
df['taglist']=df['tags'].apply(lambda x: ast.literal_eval(x))

# We use the `NLABELS` most frequent tags as keyword labels we wish to
# predict:

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
# We vectorize the text samples and labels into a 2D integer
# tensors. `MAX_NUM_WORDS` is the number of different words to use as
# tokens, selected based on word frequency. `MAX_SEQUENCE_LENGTH` is
# the fixed sequence length obtained by truncating or padding the
# original sequences.

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

# Next, we split the data into a training set and a validation set.
# We use a fraction of the data specified by `VALIDATION_DATA` for
# validation.  Note that we do not use a separate test set in this
# notebook, due to the small size of the dataset.

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

# Create PyTorch DataLoaders for both data sets:

BATCH_SIZE = 16

print('Train: ', end="")
train_dataset = TensorDataset(torch.LongTensor(x_train),
                              torch.FloatTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4)
print(len(train_dataset), 'talks')

print('Validation: ', end="")
validation_dataset = TensorDataset(torch.LongTensor(x_val),
                                   torch.FloatTensor(y_val))
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=4)
print(len(validation_dataset), 'talks')

# We prepare the embedding matrix by retrieving the corresponding word
# embedding for each token in our vocabulary:

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

embedding_matrix = torch.FloatTensor(embedding_matrix)
print('Shape of embedding matrix:', embedding_matrix.shape)

# ### Initialization

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embedding_matrix,
                                                  freeze=True)
        self.conv1 = nn.Conv1d(100, 128, 5)
        self.pool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(128, 128, 5)
        self.pool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(128, 128, 5)
        self.pool3 = nn.MaxPool1d(35)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, NLABELS)

    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1,2)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = Net().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

print(model)

# ### Learning

def train(epoch):
    # Set model to training mode
    model.train()
    epoch_loss = 0.

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):

        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)
    
        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        epoch_loss += loss.data.item()

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()

    epoch_loss /= len(train_loader)
    print('Train Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss))

    if log is not None:
        log.add_scalar('loss', epoch_loss, epoch-1)

def evaluate(epoch=None):
    model.eval()
    loss, correct = 0, 0
    pred_vector = torch.FloatTensor()
    pred_vector = pred_vector.to(device)
    
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss += criterion(output, target).data.item()

        pred = output.data
        pred_vector = torch.cat((pred_vector, pred))

    loss /= len(validation_loader)
    
    print('Validation Epoch: {}, Loss: {:.4f}'.format(epoch, loss))

    if log is not None and epoch is not None:
        log.add_scalar('val_loss', loss, epoch-1)

    return np.array(pred_vector.cpu())

epochs = 20

for epoch in range(1, epochs + 1):
    train(epoch)
    with torch.no_grad():
        evaluate(epoch)

# ### Inference
# 
# To further analyze the results, we can produce the actual
# predictions for the validation data.

with torch.no_grad():
    predictions = evaluate(None)

# The selected threshold controls the number of label predictions
# we'll make:

threshold = 0.5
print('Label prediction threshold:', threshold)

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
print()

# Precision, recall, the F1 measure, and NDCG (normalized discounted
# cumulative gain) after k returned labels are common performance
# metrics for multi-label classification:

def dcg_at_k(vals, k):
    res = 0
    for i in range(k):
        res += vals[i][1] / np.log2(i + 2)
    return res

def scores_at_k(truevals, predvals, k):
    precision_at_k, recall_at_k, f1score_at_k, ndcg_at_k = 0, 0, 0, 0

    for j in range(len(truevals)): 
        z = list(zip(predvals[j], truevals[j]))
        sorted_z = sorted(z, reverse=True, key=lambda tup: tup[0])
        opt_z = sorted(z, reverse=True, key=lambda tup: tup[1])
        truesum = 0
        for i in range(k):
            truesum += sorted_z[i][1]
        pr = truesum / k
        rc = truesum / np.sum(truevals[0])
        if truesum>0:
            f1score_at_k += 2*((pr*rc)/(pr+rc))
        precision_at_k += pr
        recall_at_k += rc
        cg = dcg_at_k(sorted_z, k) / (dcg_at_k(opt_z, k) + 0.00000001)
        ndcg_at_k += cg

    precision_at_k /= len(truevals)
    recall_at_k /= len(truevals)
    f1score_at_k /= len(truevals)
    ndcg_at_k /= len(truevals)
    
    print('Precision@{0} : {1:.2f}'.format(k, precision_at_k))
    print('Recall@{0}    : {1:.2f}'.format(k, recall_at_k))
    print('F1@{0}        : {1:.2f}'.format(k, f1score_at_k))
    print('NDCG@{0}      : {1:.2f}'.format(k, ndcg_at_k))

scores_at_k(y_val, predictions, 5)
