_base_ = [
    '../_base_/datasets/pws.py', '../_base_/models/fcn_r50-d8.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(num_classes=2,
                    loss_decode=[
    dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
    dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)],
                     ),
    auxiliary_head=dict(num_classes=2,
                        loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ]))


# norm_cfg = dict(type='SyncBN', requires_grad=True)
# model = dict(
#     type='EncoderDecoder',
#     backbone=dict(type='UNext'),
#     decode_head=dict(
#         type='SimpleHead',
#         num_classes=2,
#         in_channels=16,
#         channels=16,
#         in_index=-1,
#         dropout_ratio=0.1,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)]),

#     # auxiliary_head=dict(
#     #     type='FCNHead',
#     #     in_channels=1024,
#     #     in_index=2,
#     #     channels=256,
#     #     num_convs=1,
#     #     concat_input=False,
#     #     dropout_ratio=0.1,
#     #     num_classes=19,
#     #     norm_cfg=norm_cfg,
#     #     align_corners=False,
#     #     loss_decode=dict(
#     #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))