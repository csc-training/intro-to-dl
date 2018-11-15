#!/bin/bash
#SBATCH -A SNIC2018-5-131
#SBATCH -N 1 --gres=gpu:k80:1 -t 1:00:00 --mem=32G

#module load GCC/6.4.0-2.28  CUDA/9.0.176  OpenMPI/2.1.1
#module load Keras/2.1.3-Python-3.6.3
#module load Pillow/5.0.0-Python-3.6.3
# module purge
# module load GCC/6.4.0-2.28 OpenMPI/2.1.2
# module load Keras/2.2.0-Python-3.6.4
module purge
module load GCC/7.3.0-2.30  CUDA/9.2.88  OpenMPI/3.1.1
module load TensorFlow/1.10.1-Python-2.7.15 Keras/2.2.2-Python-2.7.15
module load Pillow/5.3.0-Python-2.7.15 h5py/2.8.0-Python-2.7.15
#module load scikit-learn/0.20.0-Python-2.7.15
module list

export KERAS_BACKEND=tensorflow

set -xv

date
hostname
nvidia-smi

python2.7 $*

date
