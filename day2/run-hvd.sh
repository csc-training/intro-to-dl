#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2003959
#SBATCH --reservation=dlintro

module load tensorflow/nvidia-20.07-tf2-py3
module list

export DATADIR=/scratch/project_2003959/data
export KERAS_HOME=/scratch/project_2003959/keras-cache

set -xv
srun python3 $*
