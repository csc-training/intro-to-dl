# Exercise 8

In this exercise, we try using multiple GPUs.

We have prepared a few examples where one of the earlier exercises
have been converted to using either DataParallel (DP) or
DistributedDataParallel (DDP).

- `pytorch_dvc_cnn_simple_multigpu.py` which implements PyTorch DP on
  the simple dogs-vs-cats CNN. You can run this with the
  `run-2gpus.sh` to run this with 2 GPUs.
  
- `pytorch_dvc_cnn_pretrained_multigpu.py` which implements PyTorch
  DDP on the pre-trained CNN for cats-vs-dogs. You can try this with
  the `run-2gpus-torchrun.sh` script.
  
- `pytorch_imdb_gpt_multigpu.py` which implements PyTorch DDP with the
  Hugging Face trainer (basically it works out-of-the-box, we've just
  added some things for nice printing). Use `run-2gpus-torchrun.sh`.

Run these scripts and then convert them to run on 4 GPUs as
well. This should only require changing the run-scripts, not the
Python scripts.

- Can you see any speed improvement between using 1, 2 or 4 GPUs?
- Do you get the same accuracy?
- Consider per-GPU batch size vs effective batch size. (Hint: with DDP you can check number of GPUs with `dist.get_world_size()`)

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
