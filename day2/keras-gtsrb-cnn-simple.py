# coding: utf-8

# # Traffic sign classification with CNNs
#
# In this script, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of traffic signs from [The German
# Traffic Sign Recognition Benchmark]
# (http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) using
# Keras (version $\ge$ 2 is required). This script is largely based on
# the blog post [Building powerful image classification models using
# very little data]
# (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
# by François Chollet.
#
# **Note that using a GPU with this script is highly recommended.**
#
# First, the needed imports. Keras tells us which backend (Theano,
# Tensorflow, CNTK) it will be using.

import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

# If we are using TensorFlow as the backend, we can use TensorBoard to
# visualize our progress during training.

if K.backend() == "tensorflow":
    import tensorflow as tf
    from keras.callbacks import TensorBoard
    import datetime
    logdir = os.path.join(os.getcwd(), "logs",
                          "gtsrb-simple-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    try:
        os.makedirs(logdir)
        callbacks = [TensorBoard(log_dir=logdir)]
    except FileExistsError:
        callbacks = None
else:
    callbacks = None

# ## Data
#
# The training dataset consists of 5535 images of traffic signs of
# varying size. There are 43 different types of traffic signs.
#
# The validation and test sets consist of 999 and 12630 images,
# respectively.

datapath = "/cfs/klemming/scratch/m/mvsjober/data/gtsrb/train-5535"
(nimages_train, nimages_validation, nimages_test) = (5535, 999, 12630)

# ### Data augmentation
#
# First, we'll resize all training and validation images to a fized size. 
#
# Then, to make the most of our limited number of training examples,
# we'll apply random transformations to them each time we are looping
# over them. This way, we "augment" our training dataset to contain
# more data. There are various transformations readily available in
# Keras, see [ImageDataGenerator]
# (https://keras.io/preprocessing/image/) for more information.

# MobileNet
input_image_size = (128, 128)

# VGG16
# input_image_size = (75, 75)

datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        #rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        horizontal_flip=False)

noopgen = ImageDataGenerator(rescale=1./255)

# Let's put a couple of training images with the augmentation to a
# TensorBoard event file.

augm_generator = datagen.flow_from_directory(
    datapath+'/train',
    target_size=input_image_size,
    batch_size=10)

for batch, _ in augm_generator:
    break

if K.backend() == "tensorflow":
    imgs = tf.convert_to_tensor(batch)
    summary_op = tf.summary.image("augmented", imgs, max_outputs=10)
    with tf.Session() as sess:
        summary = sess.run(summary_op)
        writer = tf.summary.FileWriter(logdir)
        writer.add_summary(summary)
        writer.close()

# ### Data loaders
#
# Let's now define our real data loaders for training and validation data.

batch_size = 50

print('Train: ', end="")
train_generator = datagen.flow_from_directory(
    datapath+'/train',
    target_size=input_image_size,
    batch_size=batch_size)

print('Validation: ', end="")
validation_generator = noopgen.flow_from_directory(
    datapath+'/validation',
    target_size=input_image_size,
    batch_size=batch_size)

# Similarly as with MNIST digits, we can start from scratch and train
# a CNN for the classification task. However, due to the small number
# of training images, a large network will easily overfit, regardless
# of the data augmentation.
#
# ### Initialization

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_image_size+(3,),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

# ### Learning

epochs = 10
workers = 4
use_multiprocessing = False

print('Training for', epochs, 'epochs with', workers,
      'workers, use_multiprocessing is', use_multiprocessing)

history = model.fit_generator(train_generator,
                              steps_per_epoch=nimages_train // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=nimages_validation // batch_size,
                              verbose=2, callbacks=callbacks,
                              use_multiprocessing=use_multiprocessing,
                              workers=workers)

fname = "gtsrb-small-cnn.h5"
print('Saving model to', fname)
model.save(fname)
