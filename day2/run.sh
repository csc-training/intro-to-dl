#!/bin/bash
#SBATCH -A project_2002238 -c 10 -p gpu --gres=gpu:v100:1 -t 1:00:00 --mem=64G
#xxx SBATCH --reservation dlintro

module load tensorflow/2.0.0
module list

export DATADIR=/scratch/project_2002238/data

set -xv
srun python3.7 $*
