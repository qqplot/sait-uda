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

# load_from = 'work_dirs/local-basic/230925_1637_flatHR2fishHR_mic_hrda_s2_dae5f/latest.pth'
# load_from = 'pretrained/latest.pth'
load_from = 'pretrained/segformer.b4.1024x1024.city.160k.pth'


resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True


