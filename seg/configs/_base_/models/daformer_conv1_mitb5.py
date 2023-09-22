# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# This is the same as SegFormer but with 256 embed_dims
# SegF. with C_e=256 in Tab. 7

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        # num_classes=19,
        num_classes=13,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='conv',
                kernel_size=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg),
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            # 13개 클래스만
            class_weight=[1.0110, # Road
                          1.0065, # Sidewalk
                          1.0030, # Construction
                          1.0865, # Fence
                          1.0507, # Pole
                          1.0945, # Traffic Light
                          1.1089, # Traffic Sign
                          0.9539, # Nature
                          0.866, # Sky
                          1.1116, # Person
                          1.0529, # Rider
                          1.0000, # Car
                          0.8373, # Background
                          ]
                        #   ,0.9539, 0.918, 0.9037, 0.8786, 0.866, 0.8373]         
            )),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


