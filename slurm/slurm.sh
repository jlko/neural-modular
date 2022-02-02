#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

#SBATCH --job-name="neural-net-template"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs


/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f envs/environment.yaml
source /scratch-ssd/oatml/miniconda3/bin/activate neural


srun "${@}"

# sbatch slurm/slurm.sh python main.py
