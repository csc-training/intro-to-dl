#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=project_465000512

module purge
module load tensorflow/2.8
module list

export DATADIR=/scratch/project_465000512/data
export KERAS_HOME=/scratch/project_465000512/keras-cache
export TRANSFORMERS_CACHE=/scratch/project_465000512/transformers-cache

set -xv
python3 $*
