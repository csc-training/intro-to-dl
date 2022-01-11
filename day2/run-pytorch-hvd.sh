#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:v100:2,nvme:100
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2005299
#SBATCH --reservation=dlintro

module purge
module load pytorch
module list

export DATADIR=/scratch/project_2005299/data
export TORCH_HOME=/scratch/project_2005299/torch-cache

set -xv
srun python3 $*
