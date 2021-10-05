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


def load_dataset(ds):
    if ds == 'mnist':
        from tensorflow.keras.datasets import mnist
        return mnist.load_data(), 10
    elif ds == 'fashion-mnist':
        from tensorflow.keras.datasets import fashion_mnist
        return fashion_mnist.load_data(), 10
    elif ds == 'cifar10':
        from tensorflow.keras.datasets import cifar10
        return cifar10.load_data(), 10
    elif ds == 'cifar100':
        from tensorflow.keras.datasets import cifar100
        return cifar100.load_data(), 100
    else:
        print('ERROR: Unknown dataset specified:', args.dataset)
        sys.exit(1)
    return


def train(config):
    data, nb_classes = load_dataset(config['dataset'])
    (X_train, y_train), (X_test, y_test) = data
    image_shape = X_train.shape[1:]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # one-hot encoding:
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    print()
    print(config['dataset'].upper(), 'dataset loaded: train:', len(X_train),
          'test:', len(X_test))
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('Y_train:', Y_train.shape)


    # Multi-layer perceptron (MLP) network

    inputs = keras.Input(shape=image_shape)
    x = layers.Flatten()(inputs)

    hidden1 = config['hidden1']
    hidden2 = config['hidden2']
    dropout = config['dropout']

    x = layers.Dense(units=hidden1, activation='relu')(x)
    if dropout != 0:
        x = layers.Dropout(rate=dropout)(x)

    if hidden2 > 0:
        x = layers.Dense(units=hidden2, activation='relu')(x)
        if dropout != 0:
            x = layers.Dropout(rate=dropout)(x)

    # The last layer needs to be like this:
    outputs = layers.Dense(units=nb_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="mlp_model")

    opt = keras.optimizers.Adam(learning_rate=config['lr'])

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())

    # We'll use TensorBoard to visualize our progress during training.
    logdir = os.path.join(os.getcwd(), "logs",
                          "images-mlp-{}-{}".format(
                              datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                              '-'.join(['{}:{}'.format(k, v) for k, v in config.items()])))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    callbacks = [TensorBoard(log_dir=logdir),
                 hp.KerasCallback(logdir, config)]

    then = datetime.now()
    history = model.fit(X_train, Y_train,
                        epochs=config['epochs'],
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=2,
                        validation_split=0.2)
    print('Training duration:', datetime.now()-then)

    # Inference
    if args.eval:
        scores = model.evaluate(X_test, Y_test, verbose=2)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'fashion-mnist',
                                              'cifar10', 'cifar100'],
                        default='mnist')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--hidden1', default=50, type=int,
                        help='Number of nodes in the first hidden layer')
    parser.add_argument('--hidden2', default=0, type=int,
                        help='Number of nodes in the second hidden layer, '
                        'zero means no second layer')
    parser.add_argument('--dropout', default=0, type=float,
                        help='Dropout rate to be applied after each hidden '
                        'layer. Zero means no dropout.')
    parser.add_argument('--eval', action='store_true',
                        help='Enable evaluation on test set after training')
    args = parser.parse_args()

    train(vars(args))
