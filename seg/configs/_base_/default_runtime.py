# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add img_interval

# yapf:disable
log_config = dict(
    interval=50,
    img_interval=500,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
# load_from = 'pretrained/iter_14000.pth'
# load_from = 'work_dirs/local-basic/230914_0034_flatHR2fishHR_mic_hrda_s2_131b9/latest.pth'
# load_from = 'work_dirs/local-fish2fish/230916_0932_flatHR2fishHR_mic_hrda_s2_a9506/latest.pth'
# load_from = 'work_dirs/local-fish2fish/230917_1800_flatHR2fishHR_mic_hrda_s2_8ffaa/latest.pth'
load_from = 'work_dirs/local-fish2fish/230917_1800_flatHR2fishHR_mic_hrda_s2_8ffaa/latest.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True


