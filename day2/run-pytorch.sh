#!/bin/bash
#SBATCH -A project_2002238 -c 10 -p gpu --gres=gpu:v100:1,nvme:100 -t 1:00:00 --mem=64G
# # # ##SBATCH --reservation dlintro

module load pytorch/1.3.0
module list

set -xv
echo $TMPDIR $LOCAL_SCRATCH
export TMPDIR=$LOCAL_SCRATCH
python3.7 $*
