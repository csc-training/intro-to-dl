# Taito-GPU

## Exercise sessions

### Exercise 4

Image classification: dogs vs. cats.

* *keras-dvc-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *keras-dvc-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *keras-dvc-cnn-evaluate.py*: Dogs vs. cats evaluation with test set

Text categorization: 20 newsgroups.

* *keras-20ng-rnn.py*: 20 newsgroups classification with a RNN

## Setup

We will use Taito-GPU in groups for Exercise 4. 

1. Login to Taito-GPU using a training account (or your own CSC account):

        ssh -l trainingxxx taito-gpu.csc.fi
        
2. Set up the module environment:

        module purge
        module load python-env/3.5.3-ml
    
3. Clone and cd to the exercise repository:

        git clone https://github.com/csc-training/intro-to-dl.git -b hidata2019
        cd intro-to-dl/taito-gpu

## Edit and submit jobs

1. Edit and submit jobs:

        nano keras-test.py  # or substitute with your favorite text editor
        sbatch run.sh keras-test.py  # when using a training account
        sbatch run-nores.sh keras-test.py  # when using own CSC account
        
    You can also specify additional command line arguments, e.g.:
    
        sbatch run.sh keras-dvc-rcnn-evaluate.py dvc-small-cnn.h5

2. See the status of *your jobs* or *the queue you are using*:

        squeue -l -u trainingxxx
        squeue -l -p gpu

3. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out

4. Repeat until you are happy with the results.

## Optional: TensorBoard

1. Login again in a second terminal window to Taito-GPU, with SSH port forwarding:

        ssh -l trainingxxx -L PORT:localhost:PORT taito-gpu.csc.fi
        
   Replace `PORT` with a freely selectable port number (>1023). By default, TensorBoard uses the port 6006, but **select a different port** to avoid overlaps. 

2. Set up the module environment and start the TensorBoard server:

        module purge
        module load python-env/3.5.3-ml
        tensorboard --logdir=intro-to-dl/taito-gpu/logs --port=PORT

    To access TensorBoard, point your web browser to *localhost:PORT* .
