# Exercise 8

In this exercise, we try using multiple GPUs.

We have prepared a few examples where one of the earlier exercises
have been converted to using DistributedDataParallel (DDP).

- `pytorch_dvc_cnn_pretrained_multigpu.py` which implements PyTorch
  DDP on the pre-trained CNN for cats-vs-dogs. You can try this with
  the `run-2gpus.sh` script.
  
- `pytorch_imdb_gpt_multigpu.py` which implements PyTorch DDP with the
  Hugging Face trainer. Use `run-2gpus.sh`.

## Task 1

Compare the Python code to see what changes were made to make them work on
multiple GPUs. For example, a command like this could be used:

```bash
diff -W 180 -y pytorch_dvc_cnn_pretrained.py pytorch_dvc_cnn_pretrained_multigpu.py | less
```
(navigate with PgUp/PgDn and press 'q' to quit the view).

## Task 2

Run these scripts.

- Can you see any speed improvement when going from 1 to 2 GPUs?
- Do you get the same accuracy?
- Consider per-GPU batch size vs effective batch size. (Hint: with DDP you can check number of GPUs with `dist.get_world_size()`)

You can check if your runs are actually using multiple GPUs with the
`rocm-smi` command. Check the `JOBID` of your running job with `squeue
--me`, then run (replacing JOBID with the real number):

    srun --overlap --pty --jobid=JOBID bash

This opens a new shell session in the same machine as your job. Here
you can check your processes with `top` or the state of the GPUs with
`rocm-smi`. A useful command to follow GPU usage is:

    watch rocm-smi
    
It will update every 2 seconds. It should show values above 0% for the
GPU% column for all the GPUs you intend to use. Press Ctrl-C to exit
this view.

## Task 3

Try again with 8 GPUs (a full LUMI node) using the `run-8gpus.sh` and compare run time with 1 and 2 GPUs.
