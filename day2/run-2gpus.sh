#!/bin/bash
#SBATCH --account=project_462000699
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --reservation=PDL_GPU

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache
export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns

set -xv
python3 $*
