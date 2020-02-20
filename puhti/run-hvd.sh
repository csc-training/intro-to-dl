#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2002586
#SBATCH --reservation=dlintro

module load tensorflow/2.0.0-hvd
module list

export DATADIR=/scratch/project_2002586/data

set -xv
srun python3.7 $*

