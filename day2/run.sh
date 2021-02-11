#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH -A SNIC2021-7-1

module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 TensorFlow/2.3.1-Python-3.7.4
module load scikit-learn/0.21.3-Python-3.7.4
module list

tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/dogs-vs-cats.tar.gz -C $TMPDIR
tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/gtsrb.tar.gz -C $TMPDIR
cp /cephyr/NOBACKUP/Datasets/Practical_DL/20_newsgroup.zip $TMPDIR
cp -r /cephyr/NOBACKUP/Datasets/Practical_DL/glove.6B $TMPDIR
#unzip /cephyr/NOBACKUP/Datasets/Practical_DL/alien-vs-predator-images.zip -C $TMPDIR

export DATADIR=$TMPDIR

set -xv

mpirun python3 $*
