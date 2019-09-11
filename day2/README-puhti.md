# Day 2

## Exercise sessions

### Exercise 5

Image classification: dogs vs. cats; traffic signs.

#### TF2/Keras

* *tf2-dvc-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *tf2-dvc-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *tf2-dvc-cnn-evaluate.py*: Evaluate a trained CNN with test data
* *tf2-gtsrb-cnn-simple.py*: Traffic signs with a CNN trained from scratch
* *tf2-gtsrb-cnn-pretrained.py*: Traffic signs with a pre-trained CNN
* *tf2-gtsrb-cnn-evaluate.py*: Evaluate a trained CNN with test data

#### PyTorch

The PyTorch scripts have a slightly different setup:

* *pytorch_dvc_cnn_simple.py*: Dogs vs cats with a CNN trained from scratch
* *pytorch_dvc_cnn_pretrained.py*: Dogs vs cats with a pre-trained CNN
* *pytorch_dvc_cnn.py*: Common functions for Dogs vs cats (don't run this one directly)
* *pytorch_gtsrb_cnn_simple.py*: Traffic signs with a CNN trained from scratch
* *pytorch_gtsrb_cnn_pretrained.py*: Traffic signs with a pre-trained CNN
* *pytorch_gtsrb_cnn.py*:  Common functions for Traffic signs (don't run this one directly)

To evaluate on the test set run with the `--test` option, e.g. `sbatch run.sh pytorch_dvc_cnn_simple.py --test` 

#### Extracurricular 1:

Dogs vs. cats with data in TFRecord format: 

* *tf2-dvc_tfr-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *tf2-dvc_tfr-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *tf2-dvc_tfr-cnn-evaluate.py*: Evaluate a trained CNN with test data

#### Extracurricular 2:

There is another, small dataset `avp`, of aliens and predators. Modify dogs vs. cats to classify between them.  

### Exercise 6

Text categorization: 20 newsgroups.

#### Keras

* *keras-20ng-cnn.py*: 20 newsgroups classification with a 1D-CNN
* *keras-20ng-rnn.py*: 20 newsgroups classification with a RNN

#### PyTorch

20 newsgroups PyTorch scripts do not work on Puhti yet!

* <s>*pytorch_20ng_cnn.py*: 20 newsgroups classification with a 1D-CNN</s>
* <s>*pytorch_20ng_rnn.py*: 20 newsgroups classification with a RNN</s>

#### PyTorch / BERT

* *pytorch_20ng_bert.py*: 20 newsgroups classification with BERT pretraining

### Exercise 7

Convert a script or scripts from Exercise 5 or 6 to use multiple GPUs.

* Do you get improvements in speed?
* Do you get the same accuracy than with a single GPU?

#### Extracurricular:

1. First copy training data to local SSD on the compute node and read it from there
   in your script.  On Puhti request for local storage in your Slurm script and copy data to compute node `$LOCAL_SCRATCH`. See https://docs.csc.fi/#computing/running/creating-job-scripts/#local-storage for more information
2. Horovod is not working in Puhti yet (with TF2 or PyTorch)! <s>Experiment with Horovod to implement multi-GPU training. See [run-hvd.sh](run-hvd.sh) and [keras-dvc-cnn-simple-hvd.py](keras-dvc-cnn-simple-hvd.py), or 
[pytorch_dvc_cnn_simple_hvd.py](pytorch_dvc_cnn_simple_hvd.py).</s>

## Setup

1. Login to Puhti using a training account (or your own CSC account):

        ssh -l trainingxxx puhti.csc.fi
        
2. Set up the module environment:

        module purge
        module load tensorflow/2.0.0

   or for PyTorch:
   
        module purge
        module load pytorch/1.2.0

3. Clone and cd to the exercise repository:

        git clone https://github.com/csc-training/intro-to-dl.git
        cd intro-to-dl/day2

## Edit and submit jobs

1. Edit and submit jobs:

        nano tf2-test.py  # or substitute with your favorite text editor
        sbatch run-puhti-tf2.sh tf2-test.py  # when using a training account

   There is a separate slurm script for PyTorch, e.g.:
   
        sbatch run-puhti-pytorch.sh pytorch_dvc_cnn_simple.py

   You can also specify additional command line arguments, e.g.

        sbatch run-puhti-tf2.sh tf2-dvc-cnn-evaluate.py dvc-cnn-simple.h5

2. See the status of your jobs or the queue you are using:

        squeue -l -u trainingxxx
        squeue -l -p gpu

3. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out

7. Go to 4 until you are happy with the results.

## Optional: TensorBoard

1. Login again in a second terminal window to Puhti with SSH port forwarding:

        ssh -l trainingxxx -L PORT:localhost:PORT puhti.csc.fi
        
   Replace `PORT` with a freely selectable port number (>1023). By default, TensorBoard uses the port 6006, but **select a different port** to avoid overlaps. 

2. Set up the module environment and start the TensorBoard server:

        module purge
        module load tensorflow/2.0.0
        tensorboard --logdir=intro-to-dl/day2/logs --port=PORT

    To access TensorBoard, point your web browser to *localhost:PORT* .
