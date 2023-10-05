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


###
exp_name = None # your exp_name with row resolution
load_from = f'work_dirs/local-low/{exp_name}/latest.pth'

resume_from = None
workflow = [('train', 2)]
cudnn_benchmark = True


