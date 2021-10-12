#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --account=project_2005006
#SBATCH --reservation=dlintro

module load tensorflow/2.6
module list

export DATADIR=/scratch/project_2005006/data
export KERAS_HOME=/scratch/project_2005006/keras-cache
export TRANSFORMERS_CACHE=/scratch/project_2005006/transformers-cache

set -xv
python3 $*
