#!/bin/bash
#SBATCH --account=project_462001095
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=1:00:00
##SBATCH --reservation=pdl_day2-no-ood

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache
export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns

set -xv
torchrun --standalone --nnodes=1 --nproc_per_node=$SLURM_GPUS_PER_NODE $*
