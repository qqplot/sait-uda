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
crop_size = (1024, 1024)
flat_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
fish_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='UDADataset',
        source=dict(
            type='FlatDataset',
            data_root='/shared/s2/lab01/dataset/sait_uda/data/',
            img_dir='train_source_image',
            ann_dir='train_source_gt',
            pipeline=flat_train_pipeline),
        target=dict(
            type='FishDataset',
            data_root='/shared/s2/lab01/dataset/sait_uda/data/',
            img_dir='train_target_image',
            ann_dir='train_target_image',
            pipeline=fish_train_pipeline)),
    val=dict(
        type='FlatDataset',
        data_root='/shared/s2/lab01/dataset/sait_uda/data/',
        img_dir='val_source_image',
        ann_dir='val_source_gt',
        pipeline=test_pipeline),
    test=dict(
        type='FishDataset',
        data_root='/shared/s2/lab01/dataset/fish/',
        img_dir='train_source_image_blend',
        ann_dir='train_source_gt_blend',
        pipeline=test_pipeline))
