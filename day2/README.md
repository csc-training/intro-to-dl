# Day 2

## Exercise sessions

### Exercise 4

Image classification: dogs vs. cats; traffic signs.

* *keras-dvc-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *keras-dvc-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *keras-gtsrb-cnn-simple.py*: Traffic signs with a CNN trained from scratch
* *keras-gtsrb-cnn-pretrained.py*: Traffic signs with a pre-trained CNN

### Exercise 5

Text categorization and labeling: 20 newsgroups; Ted talks.

* *keras-20ng-cnn.py*: 20 newsgroups classification with a 1D-CNN
* *keras-20ng-rnn.py*: 20 newsgroups classification with a RNN
* *keras-ted-cnn.py*: Ted talks labeling with a 1D-CNN
* *keras-ted-rnn.py*: Ted talks labeling with a RNN

### Exercise 6

Using multiple GPUs.  

* Do you get the same accuracy than with a single GPU?
* Do you get improvements in speed?

## Setup

### First terminal window

1. Login to Taito-GPU using a training account or your CSC account:

        ssh -l trainingxxx taito-gpu.csc.fi
        
2. Set up the module environment:

        module purge
        module load python-env/3.5.3-ml
    
3. Clone and cd to the exercise repository:

        git clone https://github.com/csc-training/intro-to-dl.git
        cd intro-to-dl/day2

4. Edit and submit jobs:

        emacs/vim/nano keras-test.py
        sbatch run-python-gpu-1h.sh keras-test.py  # when using a training account
        sbatch run-python-gpu-1h-nores.sh keras-test.py  # when using own CSC account

5. See the status of your jobs or the queue you are using:

        squeue -l -u trainingxxx
        squeue -l -p gpu

6. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out

7. Go to 4 until you are happy with the results.

### Second terminal window

1. Login to Taito-GPU with SSH port forwarding:

        ssh -l trainingxxx -L PORT:localhost:PORT taito-gpu.csc.fi
        
   Replace `PORT` with a freely selectable port number (>1023). By default, TensorBoard uses the port 6006, but **select a different port** to avoid overlaps. 

2. Set up the module environment and start the TensorBoard server:

        module purge
        module load python-env/3.5.3-ml
        tensorboard --logdir=intro-to-dl/day2/logs --port=PORT

    To access TensorBoard, point your web browser to *localhost:PORT* .
