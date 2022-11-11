#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --account=project_2006678
#SBATCH --reservation=dlintro

module load tensorflow
module list

export DATADIR=/scratch/project_2006678/data
export KERAS_HOME=/scratch/project_2006678/keras-cache
export TRANSFORMERS_CACHE=/scratch/project_2006678/transformers-cache

set -xv
python3 $*
