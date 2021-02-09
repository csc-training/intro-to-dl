#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2003959
#SBATCH --reservation=dlintro

module load pytorch/nvidia-20.11-py3
module list

export DATADIR=/scratch/project_2003959/data
export TORCH_HOME=/scratch/project_2003959/torch-cache
export TMPDIR=$LOCAL_SCRATCH

set -xv
singularity_wrapper exec python3 $*
