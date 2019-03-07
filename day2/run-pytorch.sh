#!/bin/bash
#SBATCH -N 1 -c 4 --gres=gpu:K80:1 -t 4:00:00 --mem=8G
#SBATCH -A pdc-test-2019

module load anaconda/py36/5.0.1
source activate pytorch

echo Running $*
time python $*

source deactivate
