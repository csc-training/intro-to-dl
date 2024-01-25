# Exercise 8

In this exercise, we try using multiple GPUs.

Convert a script or scripts from one of the earlier exercises to use
multiple GPUs. 

The easiest option is to work with the script
[pytorch_dvc_cnn_simple.py](pytorch_dvc_cnn_simple.py), but more
complex models may benefit more from scaling up to multiple GPUs.

- Do you get improvements in speed?
- Do you get the same accuracy as with a single GPU?

We have prepared a few examples if you don't want to try it yourself:

- `pytorch_dvc_cnn_simple_multigpu.py` which implements PyTorch
  Distributed on the simple dogs-vs-cats CNN. You can run this with
  the `run-2gpus.sh` to run this with 2 GPUs.
  
- `pytorch_dvc_cnn_pretrained_multigpu.py` which implements PyTorch
  Data Distributed on the pre-trained CNN for cats-vs-dogs. You can
  try this with the `run-2gpus-torchrun.sh` script.
  
- `pytorch_imdb_gpt_multigpu.py` which implements PyTorch Data
  Distributed with the Hugging Face trainer (basically it works
  out-of-the-box, we've just added some things for nice printing). Use
  `run-2gpus-torchrun.sh`.

You can check if your runs are actually using multiple GPUs with the
`rocm-smi` command. Check the `JOBID` of your running job with `squeue
--me`, then run (replacing JOBID with the real number):

    srun --interactive --pty --jobid=JOBID bash

This opens a new shell session in the same machine as your job. Here
you can check your processes with `top` or the state of the GPUs with
`rocm-smi`. A useful command to follow GPU usage is:

    watch rocm-smi
    
It will update every 2 seconds. It should show values above 0% for the
GPU% column for all the GPUs you intend to use. Press Ctrl-C to exit
this view.
