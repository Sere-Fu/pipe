_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/pws.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
