# Exercise 6

In this exercise, we play around with image generation.  This example
is based on the ["Denoising Diffusion Implicit Models"
tutorial](https://keras.io/examples/generative/ddim/) from the Keras
web site.

![Reverse diffusion with flowers](imgs/flowers-diffusion.gif)


All the code is in one file:

- [tf2-diff-models.py](tf2-diff-models.py)

The script needs an additional package [TensorFlow
Datasets](https://www.tensorflow.org/datasets) which isn't part of the
normal module. You can install it as follows:

    pip install --user tensorflow_datasets

## Task 1

Train the diffusion model to generate pictures of flowers:

    sbatch run.sh tf2-diff-models.py

The code uses the [Oxford Flowers 102
dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102),
and after each epoch it will produce 18 sample flowers that it has
generated. You can find these in the folder
`generated_images/oxford_flowers102`, each file is named after the
epoch number.

During training, monitor the generated images and see what kinds of
images it generates after the first epoch, and how it (hopefully)
improves over time.

**Note:** In LUMI we also need to do `pip install --user protobuf==3.20.3 tensorflow_addons` and change the `optimizer`in the `model.compile()` (see the comments in the code).

## Task 2

Modify the code to run on another dataset, for example cars or German
traffic signs (take a look at the commented out parts).
