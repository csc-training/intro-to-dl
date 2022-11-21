#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2006678
#SBATCH --reservation=dlintro

module purge
module load tensorflow/2.8
module list

export DATADIR=/scratch/project_2006678/data
export KERAS_HOME=/scratch/project_2006678/keras-cache
export TRANSFORMERS_CACHE=/scratch/project_2006678/transformers-cache

set -xv
srun python3 $*
