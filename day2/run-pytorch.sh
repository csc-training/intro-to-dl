#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH -A SNIC2021-7-1

module purge
module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 PyTorch/1.6.0-Python-3.7.4 
module load Horovod/0.20.3-PyTorch-1.6.0-Python-3.7.4
module load scikit-learn/0.21.3-Python-3.7.4

module list

tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/dogs-vs-cats.tar.gz -C $TMPDIR
tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/gtsrb.tar.gz -C $TMPDIR
tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/20_newsgroup.tar.gz -C $TMPDIR
tar zxf /cephyr/NOBACKUP/Datasets/Practical_DL/alien-vs-predator-images.zip -C $TMPDIR

export DATADIR=$TMPDIR
export TORCH_HOME=$TMPDIR/torch-cache

set -xv
mpirun python3 $*
