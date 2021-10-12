# Exercise 1

In this exercise we run a simple MLP for image classification on the
supercomputer.

## Files

The main script *tf2-images-mlp.py* can be used for several datasets and with
different hyperparameters for the MLP neural network. These can all be set with
arguments on the command line. You can check them by running:

```bash
module purge
module load tensorflow/2.6
python3 tf2-images-mlp.py  --help
```

## Tasks

### Task 1

Run 10 epochs of training for the cifar10 dataset with the following settings:

- Use default learning rate 0.001
- Hidden layer 1 with 50 neurons
- Hidden layer 2 with 50 neurons
- Dropout of 0.2 after each hidden layer
- Evaluate on test set

Run this using Slurm like this (fill in correct hyperparameters, see the output
of the `--help` run mentioned above).

```
sbatch run.sh tf2-images-mlp.py --dataset cifar10 [hyperparameter options here]
```

### Task 2

Try at least five runs with different hyperparameters. Try for example with and
without dropout, and/or varying the number of hidden nodes.

You can launch these one-by-one after each other, and the runs will run in
parallell in the cluster.


### Task 3

Run multiple hyperparameter configurations using Slurm array jobs.

We have already prepared a Slurm script and a text file `args-to-test.txt` with
12 different hyperparameter options to test.

```
sbatch --array=1-12 run-array.sh tf2-images-mlp.py args-to-test.txt
```
