import os
import cv2
from tqdm import tqdm
import numpy as np

base_path = '/shared/s2/lab01/dataset/fish/'
data_path = base_path + 'train_source_gt_blend/'
out_path = base_path + 'train_source_gt_blend_lbl12/'
files = sorted(os.listdir(data_path))


for fname in tqdm(files):
    ann_name = data_path + fname
    ann = cv2.imread(data_path + fname, cv2.IMREAD_GRAYSCALE)
    ann[ann == 255] = 12
    cv2.imwrite(out_path + fname, ann)