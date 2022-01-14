#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --account=project_2005299
#SBATCH --reservation=dlintro

module load tensorflow
module list

export DATADIR=/scratch/project_2005299/data
export KERAS_HOME=/scratch/project_2005299/keras-cache
export TRANSFORMERS_CACHE=/scratch/project_2005299/transformers-cache

set -xv
python3 $*
