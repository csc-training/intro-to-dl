#!/bin/bash
#SBATCH -c 4 -p gpu --gres=gpu:k80:1 -t 1:00:00 --mem=8G
#SBATCH --reservation dlhidata

module load python-env/3.5.3-ml
module list

set -xv

date
hostname
nvidia-smi

srun python3.5 $*

date
