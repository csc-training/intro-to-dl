# Exercise 6

In this exercise we use the [Ray Tune library][1] to perform hyperparameter
optimization for an MLP for image classification.

[1]: https://docs.ray.io/en/master/tune/index.html

## Files

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


## Tasks

### Task 1

Find the best hyperparameters for the cifar10 dataset using first random search
and then Bayesian Optimization. Do at least 10 samples each and try ASHA scheduler.

It's best to reserve the whole node for this job. There is a separate run script for that:

```bash
sbatch run-tune.sh tf2-images-mlp-tune.py --dataset cifar10 [any other options...]
```
