#!/bin/bash
#SBATCH -N 1 -n 2 -c 6 -p gpu --gres=gpu:p100:2 -t 1:00:00 --mem=32G
#SBATCH --reservation dlintro

module load python-env/3.7.4-ml
module list

set -xv
mpirun -np 2 -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib -oversubscribe \
    python3.7 $*
