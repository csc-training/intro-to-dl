#!/usr/bin/env python
# coding: utf-8

# # Image classification with MLPs
#
# Dataset and hyperparameters can be selected with command line arguments.
# Run `python3 tf2-images-mlp.py --help` to see options.

import argparse
import os
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

from tensorboard.plugins.hparams import api as hp

from distutils.version import LooseVersion as LV

print('Using Tensorflow version: {}, and Keras version: {}.'.
      format(tf.__version__, tf.keras.__version__))
assert(LV(tf.__version__) >= LV("2.0.0"))


# Parse command line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100'])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--units', default='50,50',
                    help='Number of units in the hidden layers, separated by comma. '
                    'For example --units=50,20 means two hidden layers, the first '
                    'with 50 and the second with 20 units.')
parser.add_argument('--dropout', default='0',
                    help='Dropout rate after each hidden layer, separated by comma.'
                    'If only one value is specified, it is assumed to be the same '
                    'for all layers. If nothing is given, no dropout is used.' )
args = parser.parse_args()

units = [int(x) for x in args.units.split(',')]
dropout = [float(x) for x in args.dropout.split(',')] if args.dropout else None

if dropout is None:
    dropout = [0 for x in units]
elif len(dropout) == 1:
    do = dropout[0]
    dropout = [do for x in units]

if len(dropout) != len(units):
    print('ERROR: number of layers ({}) is different from number of dropout values given ({}).'.
          format(len(units), len(dropout)))
    sys.exit(1)

                   
# Load dataset

ds = args.dataset.lower()
nb_classes = 10

if ds == 'mnist':
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    image_shape = (28, 28)
elif ds == 'fashion-mnist':
    from tensorflow.keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    image_shape = (28, 28)
elif ds == 'cifar10':
    from tensorflow.keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    image_shape = (32, 32, 3)
elif ds == 'cifar100':
    from tensorflow.keras.datasets import cifar100
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    image_shape = (32, 32, 3)
    nb_classes = 100
else:
    print('ERROR: Unknown dataset specified:', args.dataset)
    sys.exit(1)
    
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# one-hot encoding:
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

print()
print(ds.upper(), 'dataset loaded: train:', len(X_train), 'test:', len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('Y_train:', Y_train.shape)


# Multi-layer perceptron (MLP) network

inputs = keras.Input(shape=image_shape)
x = layers.Flatten()(inputs)

for i, n_units in enumerate(units):
    print("Adding layer {} with {} units".format(i+1, n_units), end='')
    x = layers.Dense(units=n_units, activation="relu")(x)
    dr = dropout[i]
    if dr != 0:
        x = layers.Dropout(rate=dr)(x)
        print(" and dropout with rate {}.".format(dr), end='')
    print()

# The last layer needs to be like this:
outputs = layers.Dense(units=nb_classes, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs,
                    name="mlp_model")

print("Setting learning_rate={}".format(args.lr))
opt = keras.optimizers.Adam(learning_rate=args.lr)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print(model.summary())

hparams = {'lr': args.lr, 'units': args.units, 'dropout': args.dropout}

# We'll use TensorBoard to visualize our progress during training.
logdir = os.path.join(os.getcwd(), "logs",
                      "mnist-mlp-{}-{}".format(
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                          '-'.join(['{}:{}'.format(k, v) for k, v in hparams.items()])))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir),
             hp.KerasCallback(logdir, hparams)]

epochs = args.epochs

then = datetime.now()

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=2)

print('Training duration:', datetime.now()-then)

# Inference
scores = model.evaluate(X_test, Y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
