#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2005006
#SBATCH --reservation=dlintro

module load pytorch/1.8
module list

export DATADIR=/scratch/project_2005006/data
export TORCH_HOME=/scratch/project_2005006/torch-cache

set -xv
python3 $*
