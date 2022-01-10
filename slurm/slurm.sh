#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

#SBATCH --job-name="deep-ensembles-bma"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export SLURM_CLUSTER_NAME=oatcloud
export LOG_DIR=/scratch-ssd/$USER/ttnn/
mkdir -p $LOG_DIR
mkdir -p logs


/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f envs/environment.yaml
source /scratch-ssd/oatml/miniconda3/bin/activate ensembles

# /scratch-ssd/oatml/miniconda3/bin/wandb login af749ccc40f0e5e416a3ddce03954275054d2fed

srun "${@}"

# sbatch slurm/slurm.sh python main.py +mnistruns=Cifar10LargeExperimentResnet18