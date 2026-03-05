#!/bin/bash
#SBATCH --account=project_462001275
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --reservation=pdl-day2-no-ood

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
srun singularity run $SIF bash -c "source /scratch/$SLURM_JOB_ACCOUNT/$USER/myvenv/bin/activate && python3 $*"

