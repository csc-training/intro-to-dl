# Exercise 7

In this exercise, we try using multiple GPUs.

## Task 1

Convert a script or scripts from [Exercise 5](Exercise_5.md) or [Exercise
6](Exercise_6.md) to use multiple GPUs.

- Do you get improvements in speed?
- Do you get the same accuracy than with a single GPU?

## Extracurricular 1

Use local storage in Puhti to speed up disk access. See
[run-lscratch.sh](run-lscratch.sh), which copies the _Dogs vs. cats_ dataset to
`$LOCAL_SCRATCH`, and try for example with
[tf2-dvc-cnn-simple.py](tf2-dvc-cnn-simple.py).

Also, see CSC's guide on [Data storage for machine
learning](https://docs.csc.fi/support/tutorials/ml-data/#fast-local-drive) for
more information.

## Extracurricular 2

Experiment with Horovod to implement multi-GPU training. For TensorFlow 2/Keras, see
[run-hvd.sh](run-hvd.sh) and
[tf2-dvc-cnn-simple-hvd.py](tf2-dvc-cnn-simple-hvd.py).

<details><summary><b>PyTorch and Horovod</b></summary>
  
For PyTorch see [run-pytorch-hvd.sh](run-pytorch-hvd.sh) and
[pytorch_dvc_cnn_simple_hvd.py](pytorch_dvc_cnn_simple_hvd.py).
  
</details>
