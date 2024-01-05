#!/bin/bash
#SBATCH --account=project_462000450
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=1:00:00
##SBATCH --reservation=dlintro

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

export DATADIR=/scratch/${SLURM_JOB_ACCOUNT}/data
export TORCH_HOME=/scratch/${SLURM_JOB_ACCOUNT}/torch-cache

set -xv
python3 $*
