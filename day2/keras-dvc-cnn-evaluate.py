# coding: utf-8

# # Dogs-vs-cats classification with CNNs

from keras.models import load_model
# from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D
# from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
# from keras import applications, optimizers

# from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

import sys

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

# ## Data
#
# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set consists of 1000 images,
# and the test set of 22000 images.

datapath = "/cfs/klemming/scratch/m/mvsjober/dogs-vs-cats/train-2000"
(nimages_train, nimages_validation, nimages_test) = (2000, 1000, 22000)

# MobileNet
input_image_size = (160, 160)

# VGG16
# input_image_size = (150, 150)

noopgen = ImageDataGenerator(rescale=1./255)

batch_size = 25

print('Test: ', end="")
test_generator = noopgen.flow_from_directory(
        datapath+'/test',
        target_size=input_image_size,
        batch_size=batch_size,
        class_mode='binary')

# ### Initialization

if len(sys.argv) < 2:
    print('ERROR: model file missing')
    sys.exit()

model = load_model(sys.argv[1])

print(model.summary())

# ### Inference

workers = 4
use_multiprocessing = False

print('Evaluating model', sys.argv[1], 'with', workers,
      'workers, use_multiprocessing is', use_multiprocessing)

scores = model.evaluate_generator(test_generator,
                                  steps=nimages_test // batch_size,
                                  use_multiprocessing=use_multiprocessing,
                                  workers=workers)

print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
