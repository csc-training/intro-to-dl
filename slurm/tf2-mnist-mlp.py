#!/usr/bin/env python
# coding: utf-8

## MNIST handwritten digits classification with MLPs

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import TensorBoard

from distutils.version import LooseVersion as LV

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print('Using Tensorflow version: {}, and Keras version: {}.'.format(tf.__version__, tf.keras.__version__))
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


## Load MNIST or Fashion-MNIST
#
# Fashion-MNIST categories
#
# Label|Description|Label|Description
# -----|-----------|-----|-----------
# 0    |T-shirt/top|5    | Sandal
# 1    |Trouser    |6    | Shirt
# 2    |Pullover   |7    | Sneaker
# 3    |Dress      |8    | Bag
# 4    |Coat       |9    | Ankle boot
# 

from tensorflow.keras.datasets import mnist, fashion_mnist

## MNIST:
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
## Fashion-MNIST:
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
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('Y_train:', Y_train.shape)


## Multi-layer perceptron (MLP) network

# Model initialization:
inputs = keras.Input(shape=(28, 28))
x = layers.Flatten()(inputs)

# A simple model:
x = layers.Dense(units=50, activation="relu")(x)
x = layers.Dropout(rate=0.2)(x)
x = layers.Dense(units=50, activation="relu")(x)
x = layers.Dropout(rate=0.2)(x)

# The last layer needs to be like this:
outputs = layers.Dense(units=10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs,
                    name="mlp_model")
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
print(model.summary())


## Learning

# We'll use TensorBoard to visualize our progress during training.

logdir = os.path.join(os.getcwd(), "logs",
                      "mnist-mlp-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

epochs = 10

from datetime import datetime
then = datetime.now()

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=2)

print('Training duration:', datetime.now()-then)


## Inference

scores = model.evaluate(X_test, Y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# from sklearn.metrics import confusion_matrix
# predictions = model.predict(X_test)

# print('Confusion matrix (rows: true classes; columns: predicted classes):'); print()
# cm=confusion_matrix(y_test, np.argmax(predictions, axis=1), labels=list(range(10)))
# print(cm); print()

# print('Classification accuracy for each class:'); print()
# for i,j in enumerate(cm.diagonal()/cm.sum(axis=1)): print("%d: %.4f" % (i,j))
