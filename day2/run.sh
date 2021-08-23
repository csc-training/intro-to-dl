#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2004846
#SBATCH --reservation=dlintro

module load tensorflow
module list

export DATADIR=/scratch/project_2004846/data

set -xv
python3 $*
