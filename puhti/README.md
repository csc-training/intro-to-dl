# Puhti exercises

## Exercise sessions

### Exercise 6

Image classification: dogs vs. cats; traffic signs.

* *tf2-dvc-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *tf2-dvc-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *tf2-dvc-cnn-evaluate.py*: Evaluate a trained CNN with test data
* *tf2-gtsrb-cnn-simple.py*: Traffic signs with a CNN trained from scratch
* *tf2-gtsrb-cnn-pretrained.py*: Traffic signs with a pre-trained CNN
* *tf2-gtsrb-cnn-evaluate.py*: Evaluate a trained CNN with test data

#### Extracurricular 1:

Dogs vs. cats with data in TFRecord format: 

* *tf2-dvc_tfr-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *tf2-dvc_tfr-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *tf2-dvc_tfr-cnn-evaluate.py*: Evaluate a trained CNN with test data

#### Extracurricular 2:

There is another, small dataset `avp`, of aliens and predators. Modify dogs vs. cats to classify between them.  

### Exercise 7

Text categorization: 20 newsgroups.

* *tf2-20ng-cnn.py*: 20 newsgroups classification with a 1D-CNN
* *tf2-20ng-rnn.py*: 20 newsgroups classification with a RNN
* *tf2-20ng-bert.py*: 20 newsgroups classification with BERT pretraining

### Bonus exercises

1. Convert a script or scripts from Exercise 6 or 7 to use multiple GPUs.
* Do you get improvements in speed?
* Do you get the same accuracy as with a single GPU?
2. Use local storage in Puhti to speed up disk access.  See [run-lscratch.sh](run-lscratch.sh), which copies the dogs-vs-cats dataset to `$LOCAL_SCRATCH`, and try for example with [tf2-dvc-cnn-simple.py](tf2-dvc-cnn-simple.py).  Also, see https://docs.csc.fi/computing/running/creating-job-scripts/#local-storage for more information.
3. Experiment with Horovod to implement multi-GPU training. See [run-hvd.sh](run-hvd.sh) and [tf2-dvc-cnn-simple-hvd.py](tf2-dvc-cnn-simple-hvd.py).

## Setup

1. Login to Puhti using a training account (or your own CSC account):

        ssh -l trainingxxx puhti.csc.fi
        
2. Set up the module environment:

        module purge
        module load tensorflow/2.0.0

3. Clone and cd to the exercise repository:

        git clone https://github.com/csc-training/intro-to-dl.git -b kamk2020
        cd intro-to-dl/day2

## Edit and submit jobs

1. Edit and submit jobs:

        nano tf2-test.py  # or substitute with your favorite text editor
        sbatch run.sh tf2-test.py  # when using a training account

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

