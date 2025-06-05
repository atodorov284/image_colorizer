#!/bin/bash
#SBATCH --job-name=vit_colorizer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=11:30:00

#SBATCH --output=/scratch/%u/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/%u/slurm_logs/%x_%j.err
cd /scratch/$USER/image_colorizer
module purge
module load Python/3.9.6-GCCcore-11.2.0
source .venv/bin/activate

python src/train.py