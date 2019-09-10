#!/bin/bash
#SBATCH -N 1 -p gputest --gres=gpu:k80:1 -t 15 --mem=8G

module load python-env/3.7.4-ml
module list

set -xv
srun python3.7 $*
