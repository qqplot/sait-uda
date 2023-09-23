import argparse

import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# now_str = '2023-09-21 10_26_58'
# base_path = '/shared/s2/lab01/result/mic/' + f'out/{now_str}/'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', 
                        type=str,
                        default='/shared/s2/lab01/result/mic/out/2023-09-23 02_35_19/'
    )
    parser.add_argument('--fast', 
                        type=bool,
                        default=True
    )    
    return parser



def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def after_mask(mask, fast=True, verbose=False):
    height, width = mask.shape
    bottom_third_start = int(height * 0.55)
    bottom_third_start2 = int(height * 0.75)
    if fast:
        bottom_third_start = bottom_third_start2
    left_threshold = int(width * 0.25)
    right_threshold = int(width * 0.75)
    sky_threshold = int(height * 0.85)
    if verbose:
        print('bottom_third_start:', bottom_third_start, '| bottom_third_start2:', bottom_third_start2)
        print('left:', left_threshold, '| right:', right_threshold, '| sky_threshold:', sky_threshold)
    sum = 0 
    for y in range(bottom_third_start, height):
        for x in range(width):
            if mask[y, x] == 8:
                sum += 1
                mask[y, x] = 0     
                continue
            else:
                if y <= bottom_third_start2:
                    continue

            if y <= sky_threshold:
                if mask[y, x] not in [0, 12]:
                    sum += 1
                    mask[y, x] = 0                
            else:
                if x >= left_threshold and x <= right_threshold:
                    if mask[y, x] not in [0, 12]:
                        sum += 1
                        mask[y, x] = 0
                else:
                    if mask[y, x] not in [0, 12]:
                        sum += 1
                        mask[y, x] = 12
    return mask, sum

def after_mask_easy(mask, verbose=False):
    height, width = mask.shape

    sky_threshold = int(height * 0.85)
    left_threshold = int(width * 0.25)
    right_threshold = int(width * 0.75)    
    if verbose:
        print('\nsky_threshold:', sky_threshold)
    sum = 0 
    for y in range(sky_threshold, height):
        for x in range(width):
            if x >= left_threshold and x <= right_threshold:
                if mask[y, x] not in [0, 12]:
                    sum += 1
                    mask[y, x] = 0
            else:
                if mask[y, x] not in [0, 12]:
                    sum += 1
                    mask[y, x] = 12


    return mask, sum, sky_threshold


if __name__ == '__main__':

    parser = get_arguments()
    args = parser.parse_args()
    print("args.base_path:", args.base_path)
    if str.strip(args.base_path)[-1] == '/':
        out_path = args.base_path[:-1] + '_corr/'
    else:
        out_path = args.base_path + '_corr/'
    createFolder(out_path)
    print("Create Folder ~ out_path:", out_path)

    files = sorted(os.listdir(args.base_path))

    changed_pixs = []
    for idx, fname in tqdm(enumerate(files), total=len(files)):
        ann_name = args.base_path + fname
        mask = cv2.imread(ann_name, cv2.IMREAD_GRAYSCALE)

        mask, sum, sky_threshold = after_mask_easy(mask)

        changed_pixs.append(sum)
        cv2.imwrite(out_path + fname, mask)

    print("out_path:", out_path)
    print("sky_threshold:", sky_threshold)
    print("Avg. # of Changed Pixels:", np.mean(changed_pixs))


        # sum = 0
        # for y in range(bottom_third_start, height):
        #     for x in range(width):
        #         if y <= sky_threshold:
        #             if mask[y, x] not in [0, 12]:
        #                 sum += 1
        #                 mask[y, x] = 0                
        #         else:
        #             if x >= left_threshold and x <= right_threshold:
        #                 if mask[y, x] not in [0, 12]:
        #                     sum += 1
        #                     mask[y, x] = 0
        #             else:
        #                 if mask[y, x] not in [0, 12]:
        #                     sum += 1
        #                     mask[y, x] = 12