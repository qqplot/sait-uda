#!/bin/bash


# 1. Make directory & Copy all Images
train_dir="data/train_source_image/*"
val_dir="data/val_source_image/*"
out_dir="data/train_source_image_all"

mkdir -p $out_dir

cp $train_dir $out_dir
cp $val_dir $out_dir

# 2. Preprocess Ground Truth
python preprocess_gt.py


# 3. Make masks
python make_mask.py


# 4. Make blended images
python run_blend.py
