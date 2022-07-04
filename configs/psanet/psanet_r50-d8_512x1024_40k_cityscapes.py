_base_ = [
    '../_base_/models/psanet_r50-d8.py', '../_base_/datasets/pws.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
	pretrained='open-mmlab://resnet50_v1c',
	backbone=dict(dilations=(1,1,1,2), strides=(1,2,2,1), with_cp=True),
	decode_head=dict(num_classes=2),
	auxiliary_head=dict(num_classes=2))


