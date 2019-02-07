
# coding: utf-8

# # Traffic sign classification with CNNs
# 
# In this script, we'll evaluate a convolutional neural network (CNN,
# ConvNet) to classify images of traffic signs from [The German
# Traffic Sign Recognition Benchmark]
# (http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) using
# Keras (version >= 2 is required).
# 
# **Note that using a GPU with this script is highly recommended.**
# 
# First, the needed imports. Keras tells us which backend (Theano,
# Tensorflow, CNTK) it will be using.

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D 
from keras.preprocessing.image import (ImageDataGenerator, array_to_img, 
                                      img_to_array, load_img)
from keras import applications, optimizers

from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

import numpy as np
import sys

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

# ## Data
# 
# The training dataset consists of 5535 images of traffic signs of
# varying size. There are 43 different types of traffic signs.
# 
# The validation and test sets consist of 999 and 12630 images,
# respectively.

datapath = "/wrk/makoskel/gtsrb/train-5535"
(nimages_train, nimages_validation, nimages_test) = (5535, 999, 12630)

input_image_size = (75, 75)

noopgen = ImageDataGenerator(rescale=1./255)

batch_size = 50

test_generator = noopgen.flow_from_directory(
        datapath+'/test',  
        target_size=input_image_size,
        batch_size=batch_size)

# ### Initialization

if len(sys.argv)<2:
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
