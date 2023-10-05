#!/bin/bash

#################### update please here ################
exp_name='231002_0409_flatHR2fishHR_mic_hrda_s2_40209' 
type_name='high'  

# default
base_path="./"
show=0
iters='latest'
after_mask=1
img_dir='test_image'
data_path='./data/'
test_path="${data_path}test.csv"

config_path="pretrained/${exp_name}/${exp_name}.py"
checkpoint_path="pretrained/${exp_name}/${iters}.pth"

########################################################

# config_path="${base_path}/work_dirs/local-${type_name}/${exp_name}/${exp_name}.py"
# checkpoint_path="${base_path}/work_dirs/local-${type_name}/${exp_name}/${iters}.pth"

DATE=$(date +"%Y-%m-%dT%H_%M_%S")
out_path="./out/${DATE}_${exp_name}/"
show_dir="./show_image/${DATE}_${exp_name}/"
result_dir="./results"

mkdir -p $out_path
if [ ${show} -eq 1 ];then
   mkdir -p $show_dir
fi
mkdir -p $result_dir

echo "==================================================="
echo "#1. base_path: ${base_path}"
echo "#2. out_path: ${out_path}"
echo "#3. exp_name: ${exp_name}"
echo "#4. show_dir: ${show_dir}"
echo "#5. test_path: ${test_path}"
echo "#6. now_str: ${DATE}"
echo "==================================================="

python submit.py --base_path $base_path \
                      --out_path $out_path --config_path $config_path --checkpoint_path $checkpoint_path \
                      --show_dir $show_dir --show $show --test_path $test_path \
                      --iters $iters --type_name $type_name \
                      --exp_name $exp_name --now_str $DATE \
                      --result_dir $result_dir \
                      --after_mask $after_mask \
                      --img_dir $img_dir --data_path $data_path

