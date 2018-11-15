#!/bin/bash
#SBATCH -A SNIC2018-5-131
#SBATCH -N 1 --gres=gpu:k80:1 -t 1:00:00 --mem=32G
#SBATCH --reservation=snic2018-5-131

module purge
module load GCC/7.3.0-2.30  CUDA/9.2.88  OpenMPI/3.1.1
module load TensorFlow/1.10.1-Python-2.7.15 Keras/2.2.2-Python-2.7.15
module load Pillow/5.3.0-Python-2.7.15 h5py/2.8.0-Python-2.7.15
module list

export OMPI_MCA_mpi_warn_on_fork=0
export KERAS_BACKEND=tensorflow

set -xv

date
hostname
nvidia-smi

python2.7 $*

date
