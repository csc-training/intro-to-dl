#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2002238
#xSBATCH --reservation=dlintro

module load tensorflow/2.0.0
module list

export DATADIR=/scratch/project_2002238/data

set -xv
srun python3 $*
