#!/bin/bash
#SBATCH --account=project_2020307 # Project account used for computing resources allocation
#SBATCH --partition=gputest # Partition/queue to run the job (GPU partition)
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1              # One task per node; torchrun spawns the GPU processes itself
#SBATCH --cpus-per-task=14 # Number of CPU cores allocated to the task
#SBATCH --gres=gpu:gh200:2 # Number of GPUs allocated to the task
#SBATCH --mem=120G # Total RAM allocated for the job
#SBATCH --time=00:15:00 # Maximum runtime (HH:MM:SS)
#SBATCH --reservation=pdl-day2-no-ood

# --------------------------------------------------
# Clean environment and load required modules
# --------------------------------------------------

module purge # Removes all currently loaded modules to avoid conflicts
module load python-pytorch/2.10 # Load the PyTorch 2.10 environment module

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache

export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns
export TOKENIZERS_PARALLELISM=false
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

umask 002

set -xv
srun torchrun --standalone --nnodes=1 --nproc_per_node=2 $*
