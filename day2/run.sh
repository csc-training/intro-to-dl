#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2003747
# xSBATCHx --reservation=dlintro

module load tensorflow/nvidia-20.07-tf2-py3
module list

export DATADIR=/scratch/project_2003747/data

set -xv
singularity_wrapper exec python3 $*
