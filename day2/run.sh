#!/bin/bash
#SBATCH --account=project_462001275 # Project account used for computing resources allocation
#SBATCH --partition=small-g # Partition/queue to run the job (GPU partition)
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=7 # Number of CPU cores allocated to the task
#SBATCH --gpus-per-task=1 # Number of GPUs allocated to the task
#SBATCH --mem=60G # Total RAM allocated for the job
#SBATCH --time=2:00:00 # Maximum runtime (HH:MM:SS)
#SBATCH --reservation=pdl-day2-no-ood # Reservation slot being used for the job

# --------------------------------------------------
# Clean environment and load required modules
# --------------------------------------------------

module purge # Removes all currently loaded modules to avoid conflicts
module use /appl/local/laifs/modules # Adds custom module path used on LUMI systems
module load lumi-aif-singularity-bindings # Loads Singularity bindings for running AI containers

# --------------------------------------------------
# Define container to run the job
# --------------------------------------------------
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}" # Define scratch storage location

export DATADIR=$COURSE_SCRATCH/data # Directory where datasets are stored
export TORCH_HOME=$COURSE_SCRATCH/torch-cache # Cache location for PyTorch models and weights
export HF_HOME=$COURSE_SCRATCH/hf-cache # Cache location for Hugging Face models/tokenizers


export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns # Directory where MLflow experiment logs will be stored
export TOKENIZERS_PARALLELISM=false # Disables MIOpen kernel cache (used for AMD GPUs)
 
#export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
umask 002 # Ensures group-write permissions for created files

set -xv # Prints commands before executing them (useful for debugging)
srun singularity run $SIF bash -c "source /scratch/$SLURM_JOB_ACCOUNT/$USER/myvenv/bin/activate && python3 $*"

