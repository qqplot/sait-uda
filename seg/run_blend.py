import cv2
import numpy as np
import os
import random
from tqdm import tqdm

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    # 이미지 파일들이 있는 디렉토리 경로 설정
    src_image_dir = "data/train_source_image_all/" 
    trg_image_dir = "data/train_target_image/" 
    src_gt_dir = "data/train_source_gt_all/"

    output_image_dir = "data/train_source_image_random/"
    output_gt_dir = "data/train_source_gt_random/"
    createFolder(output_image_dir); createFolder(output_gt_dir)    

    # Images
    src_files = [file for file in os.listdir(src_image_dir) if file.endswith(".png")]
    trg_files = [file for file in os.listdir(trg_image_dir) if file.endswith(".png")]
    src_files = sorted(src_files)
    trg_files = sorted(trg_files)
    random_selection = random.sample(trg_files, len(src_files))
    print(f"Length - SOURCE: {len(src_files)}, TARGET: {len(trg_files)}")    

    image1 = cv2.imread(src_image_dir + src_files[0])
    height, width = image1.shape[:2]


    # Masks
    mask_paths = ['data/filtered_stddev_image_5.png',
                 'data/filtered_stddev_image_10.png',
                 'data/filtered_stddev_image_15.png',
                 'data/filtered_stddev_image_18.png',
                 'data/filtered_stddev_image_20.png',                 
                 ]
    masks, mask_invs = [], []
    for path in mask_paths:            
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height), cv2.INTER_NEAREST)
        mask[mask != 255] = 0
        mask_inv = cv2.bitwise_not(mask)
        masks.append(mask); mask_invs.append(mask_inv)
        print(np.unique(mask))

    print(image1.shape, masks[0].shape, mask_inv[0].shape)

    for idx, src_path in tqdm(enumerate(src_files), total=len(src_files)):
        
        i = random.randint(0, len(mask_paths)-1)
        mask, mask_inv = masks[i], mask_invs[i]

        # Images
        image1 = cv2.imread(src_image_dir + src_path)                
        trg_path = random_selection[idx]
        image2 = cv2.imread(trg_image_dir + trg_path)
        image2 = cv2.resize(image2, (width, height))

        foreground = cv2.bitwise_and(image1, image1, mask=mask)
        background = cv2.bitwise_and(image2, image2, mask=mask_inv)
        result_image = cv2.add(foreground, background)
        cv2.imwrite(output_image_dir + src_path, result_image.astype(np.uint8))

        # Ground Truth
        gt = cv2.imread(src_gt_dir + src_path, cv2.IMREAD_GRAYSCALE)
        result_gt = cv2.add(gt, mask_inv)
        result_gt = result_gt.astype(np.uint8)
        result_gt[result_gt == 255] = 12
        cv2.imwrite(output_gt_dir + src_path, result_gt.astype(np.uint8))





