#!/bin/bash
#SBATCH -N 1 -p gputest --gres=gpu:k80:1 -t 15 --mem=8G

module list

set -xv

date
hostname
nvidia-smi

srun python3.5 $*

date
