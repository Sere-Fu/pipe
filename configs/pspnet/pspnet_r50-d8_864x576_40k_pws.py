_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/pws.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))