#!/bin/bash
#SBATCH -N 1 -c 4 -p gpu --gres=gpu:k80:1 -t 1:00:00 --mem=8G
#SBATCH --reservation dlintro

module load python-env/3.6.3-ml
module list

set -xv
srun python3.6 $*
