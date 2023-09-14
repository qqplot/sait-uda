#!/bin/bash

#SBATCH --job-name=mic_large
#SBATCH --nodes=1             
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00           
#SBATCH --mem=100GB
#SBATCH --exclude=b[28-29]
#SBATCH --cpus-per-task=4
#SBATCH --output=./S-%x.%j.out     

eval "$(conda shell.bash hook)"
conda activate mic

# srun python run_experiments.py --config configs/mic/flatHR2fishHR_mic_hrda.py
srun python run_experiments.py --config configs/mic/flatHR2fishHR_mic_hrda_large.py