dataset_type = 'PWSDataset'
data_root = 'data/pws'
img_norm_cfg = dict(
    mean=[72.382, 88.926, 106.403], std=[81.624, 48.824, 56.572], to_rgb=True)
img_scale = (864, 576)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(864, 576)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion', hue_delta=0),
    dict(
        type='Normalize',
        mean=[72.382, 88.926, 106.403],
        std=[81.624, 48.824, 56.572],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(864, 576),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[72.382, 88.926, 106.403],
                std=[81.624, 48.824, 56.572],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type='PWSDataset',
            data_root='data/pws',
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(864, 576)),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='PhotoMetricDistortion', hue_delta=0),
                dict(
                    type='Normalize',
                    mean=[72.382, 88.926, 106.403],
                    std=[81.624, 48.824, 56.572],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='PWSDataset',
        data_root='data/pws',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(864, 576),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[72.382, 88.926, 106.403],
                        std=[81.624, 48.824, 56.572],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='PWSDataset',
        data_root='data/pws',
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(864, 576),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[72.382, 88.926, 106.403],
                        std=[81.624, 48.824, 56.572],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=400)
evaluation = dict(interval=400, metric='mIoU', pre_eval=True)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=3,
        out_indices=(0, 1, 2),
        dilations=(1, 1, 2),
        strides=(1, 2, 2),
        norm_cfg=dict(type='BN', requires_grad=True),
        contract_dilation=True,
        with_cp=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0)
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = 'pws_414_fcn_shallower'
gpu_ids = [0]
auto_resume = False
