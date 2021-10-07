#!/usr/bin/env python
# coding: utf-8

# # Image classification with MLPs, hyperparameter tuning
#
# Dataset and hyperparameters selected with ray tune.
# Run `python3 tf2-images-mlp-tune.py --help` to see options.

import argparse
import os
import sys
from datetime import datetime

from filelock import FileLock
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

import ray
from ray import tune
#from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.suggest.bayesopt import BayesOptSearch

import os
if 'SLURM_CPUS_PER_TASK' in os.environ:
    ray.init(num_cpus=int(os.environ['SLURM_CPUS_PER_TASK']))


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
    with FileLock(os.path.expanduser("~/.data.lock")):
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

    # print()
    # print(config['dataset'].upper(), 'dataset loaded: train:', len(X_train),
    #       'test:', len(X_test))
    # print('X_train:', X_train.shape)
    # print('y_train:', y_train.shape)
    # print('Y_train:', Y_train.shape)


    # Multi-layer perceptron (MLP) network

    inputs = keras.Input(shape=image_shape)
    x = layers.Flatten()(inputs)

    hidden1 = int(config['hidden1'])
    hidden2 = int(config['hidden2'])
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
    #print(model.summary())

    callbacks = [TuneReportCallback({
        "train_accuracy": "accuracy",
        "train_loss": "loss",
        "val_accuracy": "val_accuracy",
        "val_loss": "val_loss"
    })]

    history = model.fit(X_train, Y_train,
                        epochs=config['epochs'],
                        batch_size=32,
                        callbacks=callbacks,
                        validation_split=0.2,
                        verbose=0)

def run_tune(args):
    sched = ASHAScheduler(time_attr="training_iteration") if args.sched == 'asha' else None
    search_alg = BayesOptSearch() if args.samples == 'bayes' else None
    metric = "val_accuracy"

    analysis = tune.run(
        train,
        name="MLP",
        scheduler=sched,
        metric=metric,
        mode="max",
        search_alg=search_alg,
        #stop={
        #    "mean_accuracy": 0.99,
        #    "training_iteration": num_training_iterations
        #},
        num_samples=args.samples,
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        config={
            "dataset": args.dataset,
            "epochs": args.epochs,
            "dropout": tune.uniform(0.05, 0.5),
            "lr": tune.uniform(0.001, 0.1),
            "hidden1": tune.uniform(32, 512),
            "hidden2": tune.uniform(0, 128),
        })
    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best value for", metric, ':', analysis.best_result[metric])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', '-n', default=10, type=int,
                        help='Number of different configurations of hyperparameters to try')
    parser.add_argument('--dataset', choices=['mnist', 'fashion-mnist',
                                              'cifar10', 'cifar100'],
                        default='mnist', help='Select dataset (default: mnist)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--sampler', choices=['random', 'bayes'], default='random',
                        help='Method for selecting hyperparameter configurations to try')
    parser.add_argument('--sched', choices=['none', 'asha'], default='none',
                        help='Scheduler that can stop trials that perform poorly')
    args = parser.parse_args()

    run_tune(args)
