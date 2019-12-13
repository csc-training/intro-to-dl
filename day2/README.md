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

#### TF2/Keras

* *tf2-20ng-cnn.py*: 20 newsgroups classification with a 1D-CNN
* *tf2-20ng-rnn.py*: 20 newsgroups classification with a RNN

#### PyTorch

* *pytorch_20ng_cnn.py*: 20 newsgroups classification with a 1D-CNN
* *pytorch_20ng_rnn.py*: 20 newsgroups classification with a RNN

#### PyTorch / BERT

* *pytorch_20ng_bert.py*: 20 newsgroups classification with BERT pretraining

### Exercise 7

Convert a script or scripts from Exercise 5 or 6 to use multiple GPUs.

* Do you get improvements in speed?
* Do you get the same accuracy than with a single GPU?

#### Extracurricular:

1. Use local storage in Puhti to speed up disk access.  See [run-lscratch.sh](run-lscratch.sh), which copies the dogs-vs-cats dataset to `$LOCAL_SCRATCH`, and try for example with [tf2-dvc-cnn-simple.py](tf2-dvc-cnn-simple.py).  Also, see https://docs.csc.fi/#computing/running/creating-job-scripts/#local-storage for more information.
2. Experiment with Horovod to implement multi-GPU training. See [run-hvd.sh](run-hvd.sh) and [tf2-dvc-cnn-simple-hvd.py](tf2-dvc-cnn-simple-hvd.py). (Horovod is not working in Puhti with PyTorch yet.) 

## Setup

1. Login to Puhti using a training account (or your own CSC account):

        ssh -l trainingxxx puhti.csc.fi
        
2. Set up the module environment:

        module purge
        module load tensorflow/2.0.0

   or for PyTorch:
   
        module purge
        module load pytorch/1.3.0

3. Clone and cd to the exercise repository:

        git clone https://github.com/csc-training/intro-to-dl.git
        cd intro-to-dl/day2

## Edit and submit jobs

1. Edit and submit jobs:

        nano tf2-test.py  # or substitute with your favorite text editor
        sbatch run.sh tf2-test.py  # when using a training account

   There is a separate slurm script for PyTorch, e.g.:
   
        sbatch run-pytorch.sh pytorch_dvc_cnn_simple.py

   You can also specify additional command line arguments, e.g.

        sbatch run.sh tf2-dvc-cnn-evaluate.py dvc-cnn-simple.h5

2. See the status of your jobs or the queue you are using:

        squeue -l -u trainingxxx
        squeue -l -p gpu

3. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out

7. Go to 4 until you are happy with the results.

## Optional: TensorBoard

### Option 1: Use SSH port forwarding

1. Login again in a second terminal window to Puhti with SSH port forwarding:

        ssh -l trainingxxx -L PORT:localhost:PORT puhti.csc.fi
        
   Replace `PORT` with a freely selectable port number (>1023). By default, TensorBoard uses the port 6006, but **select a different port** to avoid overlaps. 

2. Set up the module environment and start the TensorBoard server:

        module purge
        module load tensorflow/1.13.1
        tensorboard --logdir=intro-to-dl/day2/logs --port=PORT

3. To access TensorBoard, point your web browser to *localhost:PORT* .

### Option 2: Use tensorboard.dev

Another option is to use https://tensorboard.dev/ , which is Google's free TensorBoard server. You need a Google account to use the server.

1. Login again in a second terminal window to Puhti:

        ssh -l trainingxxx puhti.csc.fi
        
2. Set up the module environment and start the TensorBoard server:

        module purge
        module load tensorflow/2.0.0
        tensorboard dev upload --logdir=intro-to-dl/day2/logs

3. Visit the shown Google accounts URL to authorize the application.

4. To access TensorBoard, point your web browser to the displayed tensorboard.dev URL.

5. You can delete your data with:

        tensorboard dev delete --experiment_id EXPERIMENT_ID

