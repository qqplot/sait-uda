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

load_from = 'pretrained/gtaHR2csHR_mic_hrda_650a8/iter_40000_relevant.pth'
resume_from = None

workflow = [('train', 1)]
cudnn_benchmark = True


