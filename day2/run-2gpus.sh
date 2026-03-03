#!/bin/bash
#SBATCH --account=project_2017617
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:2,nvme:10
#SBATCH --reservation=pdl-day2-no-ood

module purge
module load pytorch/2.9

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache

export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns
export TOKENIZERS_PARALLELISM=false

umask 002

set -xv
torchrun --standalone --nnodes=1 --nproc_per_node=$SLURM_GPUS_PER_NODE $*
