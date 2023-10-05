import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

base_path = 'data/'


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


if __name__ == '__main__':

    out_path = base_path + 'train_source_gt_all/'
    createFolder(out_path)

    print("================ Train data ================")
    in_path = base_path + 'train_source_gt/'
    files = sorted(os.listdir(in_path))
    changed_pixs = []
    for fname in tqdm(files):
        ann_name = in_path + fname
        mask = cv2.imread(ann_name, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12
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

    print("================ Valid data ================")
    in_path = base_path + 'val_source_gt/'
    files = sorted(os.listdir(in_path))
    changed_pixs = []
    for fname in tqdm(files):
        ann_name = in_path + fname
        mask = cv2.imread(ann_name, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12
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
    
    print("!!! DONE !!!")