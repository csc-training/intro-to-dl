#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gres=gpu:v100:4,nvme:100
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2003747
# xSBATCHx --reservation=dlintro

module purge
module load pytorch/1.3.1-hvd
module list

export DATADIR=/scratch/project_2003747/data
export TORCH_HOME=/scratch/project_2003747/torch-cache
export TMPDIR=$LOCAL_SCRATCH

set -xv
python3 $*
