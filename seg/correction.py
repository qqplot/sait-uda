import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

base_path = '/shared/s2/lab01/dataset/sait_uda/data/'
in_path = base_path + 'val_source_gt_lbl12/'
out_path = base_path + 'val_source_gt_lbl12_correct/'


if __name__ == '__main__':

    files = sorted(os.listdir(in_path))
    changed_pixs = []
    for fname in tqdm(files):
        ann_name = in_path + fname
        mask = cv2.imread(ann_name, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape
        bottom_third_start = height * 2 // 3
        sum = 0
        for y in range(bottom_third_start, height):
            for x in range(width):
                if mask[y, x] == 8:
                    sum += 1
                    mask[y, x] = 12
        changed_pixs.append(sum)
        cv2.imwrite(out_path + fname, mask)

    print("Avg. # of Changed Pixels:", np.mean(changed_pixs))