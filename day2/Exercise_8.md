# Exercise 8

In this exercise, we try using multiple GPUs.

## Task 1

We have prepared an example where the dogs vs cats classification
using a pretrained CNN model (from Exercise 5) has been converted to
using multiple GPUs with PyTorch DDP:
[`pytorch_dvc_cnn_pretrained_multigpu.py`](pytorch_dvc_cnn_pretrained_multigpu.py).

Compare the Python code to see what changes were made to make them work on
multiple GPUs. For example, a command like this could be used:

```bash
diff -W 180 -y pytorch_dvc_cnn_pretrained.py pytorch_dvc_cnn_pretrained_multigpu.py | less
```
(navigate with PgUp/PgDn and press 'q' to quit the view).

Run this example with two GPUs using the `run-2gpu.sh` script:

```bash
sbatch run-2gpus.sh pytorch_dvc_cnn_pretrained_multigpu.py
```

Can you see any speed improvement when going from 1 to 2 GPUs? You can
compare the main training time (ignore the fine-tuning part).

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

## Task 2

We have also prepared another version of the dogs vs cats example
which uses a larger training dataset (22000 images rather than
2000). Try that first with one GPU:

```bash
sbatch run.sh pytorch_dvc_cnn_pretrained_largedata.py
```

and then with 2 GPUs:

```bash
sbatch run-2gpus.sh pytorch_dvc_cnn_pretrained_largedata_multigpu.py
```

Can you now see any improvement going from 1 to 2 GPUs.

## Extra: Task 3

If you have time, try also with 8 GPUs, that is a full node of LUMI.

For the small training set:

```bash
sbatch run-8gpus.sh pytorch_dvc_cnn_pretrained_multigpu.py
```


For the large training set:

```bash
sbatch run-8gpus.sh pytorch_dvc_cnn_pretrained_largedata_multigpu.py
```

Compare training time to the 1 and 2 GPU cases.

## Extra: Task 4

The prepared scripts `pytorch_dvc_cnn_pretrained_multigpu.py` and
`pytorch_dvc_cnn_pretrained_largedata_multigpu.py` keep the per-GPU
batch size constant (weak scaling). Implement constant effective batch
size (strong scaling), does that affect the speed and accuracy?

Hint: with DDP you can check number of GPUs with
`dist.get_world_size()`.

Hint 2: if you get errors about target size being different than input
size, you can set `drop_last=True` to the data loaders. (This has to
do with the fact that as we split batches differently, the last batch
might not have enough items.)

