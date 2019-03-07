# coding: utf-8

# # Traffic sign classification with CNNs

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

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

datapath = "/cfs/klemming/scratch/m/mvsjober/data/gtsrb/train-5535"
(nimages_train, nimages_validation, nimages_test) = (5535, 999, 12630)

# MobileNet
input_image_size = (128, 128)

# VGG16
# input_image_size = (75, 75)

noopgen = ImageDataGenerator(rescale=1./255)

batch_size = 50

test_generator = noopgen.flow_from_directory(
    datapath+'/test',
    target_size=input_image_size,
    batch_size=batch_size)

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
