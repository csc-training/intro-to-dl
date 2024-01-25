#!/bin/bash
#SBATCH --account=project_462000450
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=1:00:00
##SBATCH --reservation=PDL_GPU

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache

set -xv
torchrun --standalone --nnodes=1 --nproc_per_node=8 $*
