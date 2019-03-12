# Day 2

## Exercise sessions

### Exercise 5

Image classification: dogs vs. cats; traffic signs.

#### Keras

* *keras-dvc-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *keras-dvc-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *keras-dvc-cnn-evaluate.py*: Evaluate a trained CNN with test data
* *keras-gtsrb-cnn-simple.py*: Traffic signs with a CNN trained from scratch
* *keras-gtsrb-cnn-pretrained.py*: Traffic signs with a pre-trained CNN
* *keras-gtsrb-cnn-evaluate.py*: Evaluate a trained CNN with test data

#### PyTorch

The PyTorch scripts have a slightly different setup:

* *pytorch_dvc_cnn_simple.py*: Dogs vs cats with a CNN trained from scratch
* *pytorch_dvc_cnn_pretrained.py*: Dogs vs cats with a pre-trained CNN
* *pytorch_dvc_cnn.py*: Common functions for Dogs vs cats (don't run this one directly)
* *pytorch_gtsrb_cnn_simple.py*: Traffic signs with a CNN trained from scratch
* *pytorch_gtsrb_cnn_pretrained.py*: Traffic signs with a pre-trained CNN
* *pytorch_gtsrb_cnn.py*:  Common functions for Traffic signs (don't run this one directly)

To evaluate on the test set run with the `--test` option, e.g. `sbatch run.sh pytorch_dvc_cnn_simple.py --test` 

### Exercise 6

Text categorization and labeling: 20 newsgroups; Ted talks.

#### Keras

* *keras-20ng-cnn.py*: 20 newsgroups classification with a 1D-CNN
* *keras-20ng-rnn.py*: 20 newsgroups classification with a RNN
* *keras-ted-cnn.py*: Ted talks labeling with a 1D-CNN
* *keras-ted-rnn.py*: Ted talks labeling with a RNN

#### PyTorch

* *pytorch_20ng_cnn.py*: 20 newsgroups classification with a 1D-CNN
* *pytorch_20ng_rnn.py*: 20 newsgroups classification with a RNN
* *pytorch_ted_cnn.py*: Ted talks labeling with a 1D-CNN
* *pytorch_ted_rnn.py*: Ted talks labeling with a RNN

### Exercise 7

Convert a script or scripts from Exercise 5 or 6 to use multiple GPUs.

* Do you get improvements in speed?
* Do you get the same accuracy than with a single GPU?

Extracurricular:

1. Copy training data to compute node `$TMPDIR` and read it from there
   in your script. See Section 6.5.5 in
   https://research.csc.fi/taito-gpu-running for more information.
2. Experiment with Horovod to implement multi-GPU training. See [run-hvd.sh](run-hvd.sh) and [keras-dvc-cnn-simple-hvd.py](keras-dvc-cnn-simple-hvd.py), or 
[pytorch_dvc_cnn_simple_hvd.py](pytorch_dvc_cnn_simple_hvd.py).

## Setup

1. Login to Taito-GPU using a training account (or your own CSC account):

        ssh -l trainingxxx taito-gpu.csc.fi
        
2. Set up the module environment:

        module purge
        module load python-env/3.6.3-ml
    
3. Clone and cd to the exercise repository:

        git clone https://github.com/csc-training/intro-to-dl.git
        cd intro-to-dl/day2

## Edit and submit jobs

1. Edit and submit jobs:

        nano keras-test.py  # or substitute with your favorite text editor
        sbatch run.sh keras-test.py  # when using a training account
        sbatch run-nores.sh keras-test.py  # when using own CSC account

   You can also specify additional command line arguments, e.g.

        sbatch run.sh keras-dvc-cnn-evaluate.py dvc-small-cnn.h5      

2. See the status of your jobs or the queue you are using:

        squeue -l -u trainingxxx
        squeue -l -p gpu

3. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out

7. Go to 4 until you are happy with the results.

## Optional: TensorBoard

1. Login again in a second terminal window to Taito-GPU with SSH port forwarding:

        ssh -l trainingxxx -L PORT:localhost:PORT taito-gpu.csc.fi
        
   Replace `PORT` with a freely selectable port number (>1023). By default, TensorBoard uses the port 6006, but **select a different port** to avoid overlaps. 

2. Set up the module environment and start the TensorBoard server:

        module purge
        module load python-env/3.5.3-ml
        tensorboard --logdir=intro-to-dl/day2/logs --port=PORT

    To access TensorBoard, point your web browser to *localhost:PORT* .
