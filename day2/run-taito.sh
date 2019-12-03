#!/bin/bash
#SBATCH -N 1 -c 6 -p gpu --gres=gpu:p100:1 -t 1:00:00 --mem=16G
#SBATCH --reservation dlintro

source link-models.sh

module load python-env/3.7.4-ml
module list

set -xv
srun python3.7 $*
