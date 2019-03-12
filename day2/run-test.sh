#!/bin/bash
#SBATCH -N 1 -c 4 --gres=gpu:K420:1 -t 1:00:00 --mem=8G
#SBATCH -A pdc-test-2019

module load anaconda/py36/5.0.1
source activate tensorflow1.6

echo Running $*
time python $*

rm -Rf /tmp/.keras
source deactivate
