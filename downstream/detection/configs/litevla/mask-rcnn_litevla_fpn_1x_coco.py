_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False, 
        begin=0,
        end=1000),  # 500 -> 1000
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,  # 1e-4(ours), 2e-4(poolfmr), 2e-2(r50)
        betas=(0.9, 0.999),
        weight_decay=0.05  # 5e-2(ours), 1e-4(poolfmr), 1e-4(r50)
    )
)
