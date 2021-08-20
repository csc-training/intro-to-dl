# Exercise 7

## Tasks

### Task 1

Convert a script or scripts from Exercise 5 or 6 to use multiple GPUs.

* Do you get improvements in speed?
* Do you get the same accuracy than with a single GPU?

### Extracurricular 1

Use local storage in Puhti to speed up disk access.  See [run-lscratch.sh](run-lscratch.sh), which copies the dogs-vs-cats dataset to `$LOCAL_SCRATCH`, and try for example with [tf2-dvc-cnn-simple.py](tf2-dvc-cnn-simple.py).  Also, see https://docs.csc.fi/#computing/running/creating-job-scripts/#local-storage for more information.

### Extracurricular 2

Experiment with Horovod to implement multi-GPU training. See [run-hvd.sh](run-hvd.sh) and [tf2-dvc-cnn-simple-hvd.py](tf2-dvc-cnn-simple-hvd.py) or [run-pytorch-hvd.sh](run-pytorch-hvd.sh) and [pytorch_dvc_cnn_simple_hvd.py](pytorch_dvc_cnn_simple_hvd.py) plus [pytorch_dvc_cnn_hvd.py](pytorch_dvc_cnn_hvd.py) for PyTorch.

