#!/bin/bash
#SBATCH -A project_2002238 -c 10 -p gpu --gres=gpu:v100:1,nvme:10 -t 1:00:00 --mem=64G
#xxx SBATCH --reservation dlintro

module load tensorflow/2.0.0
module list

export DATADIR=$LOCAL_SCRATCH

set -xv

tar xf /scratch/project_2002238/data/dogs-vs-cats.tar -C $LOCAL_SCRATCH

srun python3.7 $*
