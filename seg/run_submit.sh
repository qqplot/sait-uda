#!/bin/bash

#SBATCH --job-name=submit
#SBATCH --nodes=1             
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00           
#SBATCH --mem=30GB
#SBATCH --exclude=b[13,28-29]
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_log/S-%x.%j.out     

eval "$(conda shell.bash hook)"
conda activate mic

unset CUDA_VISIBLE_DEVICES

srun python submit.py
# srun python run_experiments.py --config configs/mic/flatHR2fishHR_mic_hrda.py
# srun python run_experiments.py --config configs/mic/flatHR2fishHR_mic_hrda_large.py