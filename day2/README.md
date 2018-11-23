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

1. Login to Kebnekaise using a provided training account or your existing account:

        ssh USERNAME@kebnekaise.hpc2n.umu.se

2. Create the following links to parallel file system:
        
        cd $HOME
        ln -s /pfs/nobackup$HOME pfs
        cd pfs
        mkdir .keras 
        cd $HOME 
        ln -s pfs/.keras .keras

3. Clone and cd to the exercise repository:

        cd ~/pfs
        git clone https://github.com/csc-training/intro-to-dl.git
        cd intro-to-dl/day2
        git checkout hpc2n

4. Edit and submit jobs:

        emacs/vim/nano keras-test.py
        sbatch run.sh keras-test.py        # when using the reserved nodes
        sbatch run-nores.sh keras-test.py  # when not using the reserved nodes

5. See the status of your jobs or the queue you are using:

        squeue -l -u USERNAME
        squeue -l -p gpu
        squeue -l -R snic2018-5-131

6. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out

7. Go to 4 until you are happy with the results.

### Second terminal window

1. Login to Kebnekaise with SSH port forwarding:

        ssh -L PORT:localhost:PORT USERNAME@kebnekaise.hpc2n.umu.se
        
   Replace `PORT` with a freely selectable port number (>1023). By default, TensorBoard uses the port 6006, but **select a different port** to avoid overlaps. 

2. Set up the module environment and start the TensorBoard server:

        module load GCC/7.3.0-2.30  CUDA/9.2.88  OpenMPI/3.1.1
        module load TensorFlow/1.10.1-Python-2.7.15
        tensorboard --logdir=pfs/intro-to-dl/day2/logs --port=PORT

    To access TensorBoard, point your web browser to *localhost:PORT* .
