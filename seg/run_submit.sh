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

#################### update please here ################
base_path="/home/s2/kyubyungchae/MIC/seg/"
exp_name='230925_2105_flatHR2fishHR_mic_hrda_s2_f0abe'
show=true
iters='iter_1000'
type_name="tiny"
########################################################


DATE=$(date +"%Y-%m-%dT%H_%M_%S")
config_path="${base_path}/work_dirs/local-${type_name}/${exp_name}/${exp_name}.py"
checkpoint_path="${base_path}/work_dirs/local-${type_name}/${exp_name}/${iters}.pth"
out_path="/shared/s2/lab01/result/mic/out/${DATE}_${exp_name}/"
show_dir="/shared/s2/lab01/result/mic/show_image/${DATE}_${exp_name}/"
test_path="${base_path}/test.csv"

echo "==================================================="
echo "#1. base_path: ${base_path}"
echo "#2. out_path: ${out_path}"
echo "#3. exp_name: ${exp_name}"
echo "#4. show_dir: ${show_dir}"
echo "#5. test_path: ${test_path}"
echo "#6. now_str: ${DATE}"
echo "==================================================="

srun python submit.py --base_path $base_path \
                      --out_path $out_path --config_path $config_path --checkpoint_path $checkpoint_path \
                      --show_dir $show_dir --show $show --test_path $test_path \
                      --iters $iters --type_name $type_name \
                      --exp_name $exp_name --now_str $DATE

# srun python run_experiments.py --config configs/mic/flatHR2fishHR_mic_hrda.py
# srun python run_experiments.py --config configs/mic/flatHR2fishHR_mic_hrda_large.py
