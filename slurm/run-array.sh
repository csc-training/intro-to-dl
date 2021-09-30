#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2004846
#SBATCH --reservation=dlintro

module load tensorflow
module list

export DATADIR=/scratch/project_2004846/data
export KERAS_HOME=/scratch/project_2004846/keras-cache
export TRANSFORMERS_CACHE=/scratch/project_2004846/transformers-cache

PYTHON_SCRIPT=$1
ARGFILE=$2

if [ -z "$PYTHON_SCRIPT" -o -z "$ARGFILE" ]; then
    echo "Usage: sbatch run-array.sh [python_script.py] [arguments_file.txt]"
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: Please launch this script as an array job, for example:"
    echo "sbatch --array=0-10 run-array.sh $PYTHON_SCRIPT $ARGFILE"
    exit 1
fi

# Read arguments from file from the line specified by $SLURM_ARRAY_TASK_ID
# See: https://docs.csc.fi/computing/running/array-jobs/#using-a-file-name-list-in-an-array-job

ARGS=$(sed -n ${SLURM_ARRAY_TASK_ID}p $ARGFILE)

if [ -z "$ARGS" ]; then
    echo "ERROR: No arguments on line ${SLURM_ARRAY_TASK_ID} in ${ARGFILE}."
    exit 1
fi

set -xv
python3 $PYTHON_SCRIPT $ARGS
