# Exercise 6

In this exercise we use the [Ray Tune library][1] to perform hyperparameter
optimization for an MLP for image classification.

[1]: https://docs.ray.io/en/master/tune/index.html

## Files

The main script *tf2-images-mlp-tune.py* can be used for several datasets and
you can select different samplers and schedulers with arguments on the command
line. You can check them by running:

```bash
module purge
module load tensorflow
python3 tf2-images-mlp-tune.py  --help
```

## Tasks

### Task 1

Find the best hyperparameters for the cifar10 dataset using first random search
and then Bayesian Optimization. Do at least 50 samples each and try ASHA scheduler.
