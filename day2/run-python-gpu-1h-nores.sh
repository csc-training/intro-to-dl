#!/bin/bash
#SBATCH -N 1 -p gpu --gres=gpu:p100:1 -t 1:00:00 --mem=8G

module list

set -xv

date
hostname
nvidia-smi

srun python3.5 $*

date
