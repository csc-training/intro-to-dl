#!/bin/bash
#SBATCH --account=project_462000131 # switch to the actual project
#SBATCH --partition=dev-g # switch to the actual partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=0:30:00
##SBATCH --reservation=pdl-day2-no-ood # uncomment this for the actual course

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache

export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns
export TOKENIZERS_PARALLELISM=false

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=""
umask 002

set -xv
singularity exec $SIF torchrun --standalone --nnodes=1 --nproc_per_node=$SLURM_GPUS_PER_NODE $*
