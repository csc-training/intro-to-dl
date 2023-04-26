# Exercise 6

In this exercise, we play around with image generation.  This example
is based on the ["Denoising Diffusion Implicit Models"
tutorial](https://keras.io/examples/generative/ddim/) from the Keras
web site.

All the code is in one file:

- [tf2-diff-models.py](tf2-diff-models.py)

The script needs an additional package [TensorFlow
Datasets](https://www.tensorflow.org/datasets) which isn't part of the
normal module. You can install it as follows:

    module purge
    module load tensorflow
    pip install --user tensorflow_datasets

You can run the code as:

    sbatch run.sh tf2-diff-models.py

By default it trains on the [Oxford Flowers 102
dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102),
and after each epoch it will produce 18 sample flowers that it has
generated. You can find these in the folder
`generated_images/oxford_flowers102`, each file is named after the
epoch number.

![Reverse diffusion with flowers](imgs/flowers-diffusion.gif)

You can also change the code to generate cars or German traffic signs
(take a look at the commented out parts).
