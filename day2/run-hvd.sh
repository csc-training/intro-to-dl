#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2003747
# xSBATCHx --reservation=dlintro

module load tensorflow/nvidia-20.07-tf2-py3
module list

export DATADIR=/scratch/project_2003747/data
export TRANSFORMERS_CACHE=/scratch/project_2003747/transformers-cache

set -xv
srun singularity_wrapper exec python3 $*
