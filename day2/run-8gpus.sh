#!/bin/bash
#SBATCH --account=project_462000863
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=1:00:00
#SBATCH --reservation=pdl_day2-no-ood

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache

export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns
export TOKENIZERS_PARALLELISM=false

umask 002

set -xv
torchrun --standalone --nnodes=1 --nproc_per_node=$SLURM_GPUS_PER_NODE $*
