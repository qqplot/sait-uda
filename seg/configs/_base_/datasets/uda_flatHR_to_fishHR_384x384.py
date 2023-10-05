# Obtained from: https://github.com/lhoyer/HRDA
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'FlatDataset'
data_root = '/shared/s2/lab01/dataset/sait_uda/data/train_source_image/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='GaussianBlur', prob=0.2),
    # dict(type='GaussianNoise', prob=0.2, std=10.0),
    # dict(type='RandomRotate', prob=0.2, degree=20, pad_val=0),       
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='UDADataset',
        source=dict(
            type='FlatDataset',
            data_root='data/',
            img_dir='train_source_image_random',
            ann_dir='train_source_gt_random',
            pipeline=train_pipeline),
        target=dict(
            type='FishDataset',
            data_root='data/',
            img_dir='train_target_image',
            ann_dir='train_target_image',
            pipeline=train_pipeline)),
    val=dict(
        type='FlatDataset',
        data_root='data/',
        img_dir='val_source_image',
        ann_dir='val_source_gt',
        pipeline=test_pipeline),
    test=dict(
        type='FishDataset',
        data_root='data/',
        img_dir='train_source_image',
        ann_dir='train_source_gt',
        pipeline=test_pipeline))