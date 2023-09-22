from PIL import Image
import numpy as np
import pandas as pd

from datetime import datetime
from pytz import timezone
from tqdm import tqdm

import mmcv

base_path = '/home/s2/kyubyungchae/MIC/seg/'
test_path = base_path + 'test.csv'
# now_str = '2023-09-21 10_26_58_corr'
# now_str = '2023-09-21 22_08_44_corr'
now_str = '2023-09-22 11_38_57_corr'

out_path = f'/shared/s2/lab01/result/mic/out/{now_str}/'
data_path = '/shared/s2/lab01/dataset/sait_uda/data/'
submission_path = data_path + '/sample_submission.csv'


# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

show = True

if __name__ == '__main__':
    test_img_list = pd.read_csv(test_path)
    test_img_name_lis = test_img_list.iloc[:,1]
    
    submit_outputs = []
    for i, fname in tqdm(enumerate(test_img_name_lis), total=len(test_img_name_lis)):
            pred = mmcv.imread(out_path + fname.split('/')[-1], 'grayscale')
            pred_numpy = pred.astype(np.uint8)
            pred_numpy = Image.fromarray(pred_numpy) # 이미지로 변환
            pred_numpy = pred_numpy.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
            pred_numpy = np.array(pred_numpy) # 다시 수치로 변환

            for class_id in range(12):
                class_mask = (pred_numpy == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                    mask_rle = rle_encode(class_mask)
                    submit_outputs.append(mask_rle)
                else: # 마스크가 존재하지 않는 경우 -1
                    submit_outputs.append(-1)

    submit = pd.read_csv(submission_path)
    submit['mask_rle'] = submit_outputs

    result_dir = './results'
    submit.to_csv(result_dir + '/' + now_str + ' submit.csv', index=False)

