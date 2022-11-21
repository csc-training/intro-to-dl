#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2006678
#SBATCH --reservation dlintro

module purge
module load tensorflow/2.8
module list

export DATADIR=$LOCAL_SCRATCH
export KERAS_HOME=/scratch/project_2006678/keras-cache
export TRANSFORMERS_CACHE=/scratch/project_2006678/transformers-cache

set -xv

tar xf /scratch/project_2006678/data/dogs-vs-cats.tar -C $LOCAL_SCRATCH

python3 $*
