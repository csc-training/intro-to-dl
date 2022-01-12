# Exercise X

In this exercise we use [Slurm array jobs](https://docs.csc.fi/computing/running/array-jobs/) and the [Ray Tune library][1] to perform hyperparameter
optimization for an MLP for image classification.

[1]: https://docs.ray.io/en/master/tune/index.html

## Part I: Manual jobs and array jobs

### Files

The main script *tf2-images-mlp.py* can be used for several datasets and with
different hyperparameters for the MLP neural network. These can all be set with
arguments on the command line. You can check them by running:

```bash
module purge
module load tensorflow/2.6
python3 tf2-images-mlp.py  --help
```

### Tasks

#### Task 1

Run 10 epochs of training for the cifar10 dataset with the following settings:

- Use default learning rate 0.001
- Hidden layer 1 with 50 neurons
- Hidden layer 2 with 50 neurons
- Dropout of 0.1 after each hidden layer
- Evaluate on test set

Run this using Slurm like this (fill in correct hyperparameters, see the output
of the `--help` run mentioned above).

```
sbatch run.sh tf2-images-mlp.py --dataset cifar10 [hyperparameter options here]
```

#### Task 2

Try at least five runs with different hyperparameters. Try for example with and
without dropout, and/or varying the number of hidden nodes.

You can launch these one-by-one after each other, and the runs will run in
parallel in the cluster.


#### Task 3

Run multiple hyperparameter configurations using Slurm array jobs.

We have already prepared a Slurm script and a text file `args-to-test.txt` with
12 different hyperparameter options to test.

```
sbatch --array=1-12 run-array.sh tf2-images-mlp.py args-to-test.txt
```

## Part II: Ray Tune

### Files

The main script *tf2-images-mlp-tune.py* can be used for several datasets and
you can select different samplers and schedulers with arguments on the command
line. 

```
usage: tf2-images-mlp-tune.py [-h] [--samples SAMPLES] [--dataset {mnist,fashion-mnist,cifar10,cifar100}]
                              [--epochs EPOCHS] [--sampler {random,bayes}] [--sched {none,asha}]

optional arguments:
  -h, --help            show this help message and exit
  --samples SAMPLES, -n SAMPLES
                        Number of different configurations of hyperparameters to try (default: 10)
  --dataset {mnist,fashion-mnist,cifar10,cifar100}
                        Select dataset (default: mnist)
  --epochs EPOCHS       Number of epochs to train (default: 10)
  --sampler {random,bayes}
                        Method for selecting hyperparameter configurations to try (default: random)
  --sched {none,asha}   Scheduler that can stop trials that perform poorly (default: none)
```


### Tasks

#### Task 1

Find the best hyperparameters for the cifar10 dataset using first random search
and then Bayesian Optimization. Do at least 10 samples each and try ASHA scheduler.

It's best to reserve the whole node for this job. There is a separate run script for that:

```bash
sbatch run-tune.sh tf2-images-mlp-tune.py --dataset cifar10 [any other options...]
```
