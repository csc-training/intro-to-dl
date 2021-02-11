#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=T4:2
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH -A SNIC2021-7-1

# For TensorFlow
module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 TensorFlow/2.1.0-Python-3.7.4
module load scikit-learn/0.21.3-Python-3.7.4
module load Horovod/0.19.1-TensorFlow-2.1.0-Python-3.7.4

module list

tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/dogs-vs-cats.tar.gz -C $TMPDIR
tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/gtsrb.tar.gz -C $TMPDIR
tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/20_newsgroup.tar.gz -C $TMPDIR
tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/alien-vs-predator-images.zip -C $TMPDIR

export DATADIR=$TMPDIR
export KERAS_HOME=$TMPDIR/keras-cache

set -xv

mpirun python $*
