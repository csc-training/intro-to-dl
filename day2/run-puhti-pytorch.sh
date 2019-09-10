#!/bin/bash
#SBATCH -A project_2001756 -N 1 -c 4 -p gpu --gres=gpu:v100:1 -t 1:00:00 --mem=8G

module load pytorch/1.2.0
module list

set -xv
srun python3.7 $*
