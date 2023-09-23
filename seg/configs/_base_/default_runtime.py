# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add img_interval

# yapf:disable
log_config = dict(
    interval=50,
    img_interval=1000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
# load_from = 'work_dirs/local-basic/230914_0034_flatHR2fishHR_mic_hrda_s2_131b9/iter_16000.pth'
# load_from = 'work_dirs/local-small/230919_1506_flatHR2fishHR_mic_hrda_s2_95481/iter_20000.pth'
# load_from = 'work_dirs/local-small/230920_1934_flatHR2fishHR_mic_hrda_s2_068cf/iter_20000.pth'
# load_from = 'work_dirs/local-small/230921_1428_flatHR2fishHR_mic_hrda_s2_bef7e/iter_20000.pth'  # 0.505
load_from = 'pretrained/gtaHR2csHR_mic_hrda_650a8/iter_40000_relevant.pth'

# load_from = 'work_dirs/local-cityscapes/230922_1619_flatHR2fishHR_mic_hrda_s2_4d990/iter_20000.pth'

resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True


