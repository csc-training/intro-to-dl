# Day 2 - alvis

## Exercise sessions

### Exercise 5

Image classification: [dogs vs. cats](imgs/dvc.png); [traffic signs](imgs/gtsrb-montage.png).

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

To evaluate on the test set run with the `--test` option, e.g. `sbatch run-pytorch.sh pytorch_dvc_cnn_simple.py --test` 

#### Extracurricular 1:

Dogs vs. cats with data in TFRecord format: 

* *tf2-dvc_tfr-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *tf2-dvc_tfr-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *tf2-dvc_tfr-cnn-evaluate.py*: Evaluate a trained CNN with test data

#### Extracurricular 2:

There is another, small dataset `avp`, of [aliens and predators](imgs/avp.png). Modify dogs vs. cats to classify between them.  

### Exercise 6

Text categorization: [20 newsgroups](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html).

#### TF2/Keras

* *tf2-20ng-rnn.py*: 20 newsgroups classification with a RNN
* *tf2-20ng-cnn.py*: 20 newsgroups classification with a CNN
* *tf2-20ng-bert.py*: 20 newsgroups classification with BERT finetuning

#### PyTorch

* *pytorch_20ng_rnn.py*: 20 newsgroups classification with a RNN
* *pytorch_20ng_cnn.py*: 20 newsgroups classification with a CNN
* *pytorch_20ng_bert.py*: 20 newsgroups classification with BERT finetuning

### Exercise 7

Convert a script or scripts from Exercise 5 or 6 to use multiple GPUs.

* Do you get improvements in speed?
* Do you get the same accuracy than with a single GPU?

#### Extracurricular:

1. Use local storage in Puhti to speed up disk access.  See [run-lscratch.sh](run-lscratch.sh), which copies the dogs-vs-cats dataset to `$LOCAL_SCRATCH`, and try for example with [tf2-dvc-cnn-simple.py](tf2-dvc-cnn-simple.py).  Also, see https://docs.csc.fi/#computing/running/creating-job-scripts/#local-storage for more information.
2. Experiment with Horovod to implement multi-GPU training. See [run-hvd.sh](run-hvd.sh) and [tf2-dvc-cnn-simple-hvd.py](tf2-dvc-cnn-simple-hvd.py) or [run-pytorch-hvd.sh](run-pytorch-hvd.sh) and [pytorch_dvc_cnn_simple_hvd.py](pytorch_dvc_cnn_simple_hvd.py) plus [pytorch_dvc_cnn_hvd.py](pytorch_dvc_cnn_hvd.py) for PyTorch.

## Setup

0. You need a SUPR account to access Alvis. If you don't already have one, visit
   https://supr.snic.se/ and follow the instructions. 
   Your SUPR account will be added to project SNIC2020-5-235. After being 
   added, you can request a user account on Alvis via https://supr.snic.se/account/.

1. Login to Alvis requires that you are using a Swedish university network. 
   If you are working from a university building and are using Eduroam or an ethernet connection, 
   you should be able to log in right away.
   If you are working from home you will need to set up a Virtual Private Network (VPN) service, either the 
   [Chalmers VPN service](https://it.portal.chalmers.se/itportal/NonCDAWindows/NonCDAWindows#remote) 
   or another VPN service from your university. Contact your university's IT support to get help 
   to set it up. Further information about accessing Alvis 
   is available on the [C3SE support pages](https://www.c3se.chalmers.se/documentation/connecting/).

2. Login to Alvis using your personal account:

        ssh -l <username> alvis1.c3se.chalmers.se
        
3. Set up the module environment:

        module purge
	module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 TensorFlow/2.3.1-Python-3.7.4
	module load scikit-learn/0.21.3-Python-3.7.4

   or for PyTorch:
   
        module purge
	module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 PyTorch/1.7.1-Python-3.7.4

   For Horovod with TensorFlow (note the different version of TensorFlow):

        module purge
        module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 TensorFlow/2.1.0-Python-3.7.4
        module load Horovod/0.19.1-TensorFlow-2.1.0-Python-3.7.4

   and for Horovod with PyTorch::

        module purge
	module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 PyTorch/1.6.0-Python-3.7.4
        module load Horovod/0.20.3-PyTorch-1.6.0-Python-3.7.4

4. Clone and cd to the exercise repository:

        git clone https://github.com/csc-training/intro-to-dl.git
        cd intro-to-dl/day2

## Edit and submit jobs

1. Edit and submit jobs:

        nano tf2-test.py  # or substitute with your favorite text editor
        sbatch run.sh tf2-test.py  

   There is a separate slurm script for PyTorch, e.g.:
   
        sbatch run-pytorch.sh pytorch_dvc_cnn_simple.py

   You can also specify additional command line arguments, e.g.

        sbatch run.sh tf2-dvc-cnn-evaluate.py dvc-cnn-simple.h5

2. See the status of your jobs or the queue you are using:

        squeue -u $USER

3. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out

7. Go to 4 until you are happy with the results.

## Optional: TensorBoard

You can run TensorBoard on Alvis and connect to it from your local browser. Be aware that 
Tensorboard offers no security! Anyone with the correct URL can access your session.

1. Set up the module environment:

       module purge
       module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 TensorFlow/2.3.1-Python-3.7.4

2. Start Tensorboard on a free port:

       FREE_PORT=`comm -23 <(seq "8888" "8988" | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf -n 1`
       echo "Tensorboard URL: https://proxy.c3se.chalmers.se:${FREE_PORT}/`hostname`/"
       tensorboard --path_prefix /`hostname`/ --bind_all --port $FREE_PORT --logdir=./tensorboard-log-1234

3. To access TensorBoard, point your web browser to the URL from the `echo` command in step 2
   (e.g. https://proxy.c3se.chalmers.se:8925/alvis1/)

