# Day 2

## Exercise sessions

### Exercise 1

Image classification: dogs vs. cats; traffic signs.

Slurm scripts:
* *keras-dvc-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *keras-dvc-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *keras-gtsrb-cnn-empty.py*: Traffic signs, only data loading implemented
* *keras-gtsrb-cnn-simple.py*: Traffic signs with a CNN trained from scratch
* *keras-gtsrb-cnn-pretrained.py*: Traffic signs with a pre-trained CNN

### Exercise 2

Text categorization and labeling: 20 newsgroups; Ted talks.

Slurm scripts:
* *keras-20ng-cnn.py*: 20 newsgroups classification with a 1D-CNN
* *keras-20ng-rnn.py*: 20 newsgroups classification with a RNN
* *keras-ted-cnn.py*: Ted talks labeling with a 1D-CNN
* *keras-ted-rnn.py*: Ted talks labeling with a RNN

## Setup

1. Login to Taito-GPU using a training account:

        ssh -l training0xx -X taito-gpu.csc.fi

2. Set up the module environment:

        module purge
        module load python-env/3.5.3-ml
    
3. Clone and cd to the exercise repository:

        git clone https://github.com/csc-training/intro-to-dl.git
        cd intro-to-dl/day2

4. Edit and submit jobs:

        emacs/vim/nano keras-test.py
        sbatch run-python-gpu-1h.sh keras-test.py

5. See the status of your jobs or the queue you are using:

        squeue -l -u training0xx
        squeue -l -p gpu

6. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out
        sxiv results-figure.png

7. Go to 4 until you are happy with the results.
