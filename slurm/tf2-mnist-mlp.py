#!/usr/bin/env python
# coding: utf-8

# MNIST handwritten digits classification with MLPs

import argparse
import os
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.datasets import mnist, fashion_mnist

from tensorboard.plugins.hparams import api as hp

from distutils.version import LooseVersion as LV

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

print('Using Tensorflow version: {}, and Keras version: {}.'.
      format(tf.__version__, tf.keras.__version__))
assert(LV(tf.__version__) >= LV("2.0.0"))


# Let's check if we have GPU available.

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.python.client import device_lib
    for d in device_lib.list_local_devices():
        if d.device_type == 'GPU':
            print('GPU', d.physical_device_desc)
else:
    print('No GPU, using CPU instead.')


# Parse command line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
parser.add_argument('--units', default='50,50',
                    help='Number of units in the hidden layers, separated by comma. '
                    'For example --units=50,20 means two hidden layers, the first '
                    'with 50 and the second with 20 units.')
parser.add_argument('--dropout', required=False,
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


# Fashion-MNIST categories
#
# Label|Description|Label|Description
# -----|-----------|-----|-----------
# 0    |T-shirt/top|5    | Sandal
# 1    |Trouser    |6    | Shirt
# 2    |Pullover   |7    | Sneaker
# 3    |Dress      |8    | Bag
# 4    |Coat       |9    | Ankle boot


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

nb_classes = 10

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# one-hot encoding:
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

print()
print('MNIST data loaded: train:', len(X_train), 'test:', len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('Y_train:', Y_train.shape)


# Multi-layer perceptron (MLP) network

inputs = keras.Input(shape=(28, 28))
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
outputs = layers.Dense(units=10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs,
                    name="mlp_model")

print("Setting learning_rate={}".format(args.learning_rate))
opt = keras.optimizers.Adam(learning_rate=args.learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print(model.summary())

hparams = {'lr': args.learning_rate, 'units': args.units, 'dropout': args.dropout}

# We'll use TensorBoard to visualize our progress during training.
logdir = os.path.join(os.getcwd(), "logs",
                      "mnist-mlp-" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir),
             hp.KerasCallback(logdir, hparams)]

epochs = 10

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
