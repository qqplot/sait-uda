import os.path as osp
import os

import torch

from PIL import Image
import numpy as np
import pandas as pd

from datetime import datetime
from pytz import timezone
from tqdm import tqdm

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmcv.image import tensor2imgs

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    cfg.data.val.pipeline[1]['img_scale'] = tuple(
        cfg.data.val.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)



if __name__ == '__main__':
    # Init the model from the config and the checkpoint    
    base_path = '/home/s2/kyubyungchae/MIC/seg/'

    type_name = 'small' # 'basic' 'fish2fish' 'blend' 'small'
    # exp_name = '230914_1828_flatHR2fishHR_mic_hrda_s2_022a1'
    # exp_name = '230918_0432_flatHR2fishHR_mic_hrda_s2_5c420'
    # exp_name = '230919_1506_flatHR2fishHR_mic_hrda_s2_95481' # best
    exp_name = '230920_1934_flatHR2fishHR_mic_hrda_s2_068cf' # best start
    show = True
    iters = 'latest'  # latest
    config_path = base_path + f'work_dirs/local-{type_name}/{exp_name}/{exp_name}.py'
    checkpoint_path = base_path + f'work_dirs/local-{type_name}/{exp_name}/{iters}.pth'
    test_path = base_path + '/test.csv'
    data_path = '/shared/s2/lab01/dataset/sait_uda/data/'
    submission_path = data_path + '/sample_submission.csv'

    now = datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime('%Y-%m-%d %H_%M_%S')
    out_path = '/shared/s2/lab01/result/mic/' + f'out/{now_str}/'    
    show_dir = '/shared/s2/lab01/result/mic/' + f'show_image/{now_str}/'

    createFolder(out_path)
    if show:    
        createFolder(show_dir)


    cfg = mmcv.Config.fromfile(config_path)
    cfg = update_legacy_cfg(cfg)
    cfg.data.test.test_mode = True

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])

    test_img_list = pd.read_csv(test_path)
    test_img_name_lis = test_img_list.iloc[:,1]

    cfg.data.test['data_root'] = data_path
    cfg.data.test['img_dir'] = 'test_image'
    cfg.data.test['ann_dir'] = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE 


    efficient_test = False

    model = MMDataParallel(model, device_ids=[0])


    # Inference
    # outputs = single_gpu_test(model, data_loader, show, show_dir, efficient_test, 0.5)
    model.eval()
    results = []
    dataset = data_loader.dataset

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):

        with torch.no_grad():
            result = model(return_loss=False, **data)

        ori_filename = data['img_metas'][0].data[0][0]['ori_filename']         
        if isinstance(result, list):
            pass
        else:
            result = [result]
        
        for pred in result:
            pred_numpy = pred.astype(np.uint8)
            pred_numpy = Image.fromarray(pred_numpy) # 이미지로 변환
            pred_numpy.save(f'{out_path}{ori_filename}')
            # results.append(result)
        
        if show:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                out_file = osp.join(show_dir, img_meta['ori_filename'])

                if hasattr(model.module.decode_head,
                           'debug_output_attention') and \
                        model.module.decode_head.debug_output_attention:
                    # Attention debug output
                    mmcv.imwrite(result[0] * 255, out_file)
                else:
                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=0.6)

        ######
    
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

    print("DONE!")
    print("outpath:", out_path)
    print("chk_path:", checkpoint_path)