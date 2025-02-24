_base_ = [
    './mask-rcnn_litevla_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_LITEVLA',
        version='litevla_m',
        backbone=True,
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/litevla_m.pth"  # TODO [HSC]: change to github release url later
    ),
    neck=dict(in_channels=(48, 96, 192, 384)),
)

train_dataloader = dict(batch_size=4)  # as gpus=4


# TODO [HSC]: remove when release
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='det_mask-rcnn',
            name='litevla_m',
        )
    )]
visualizer = dict(vis_backends=vis_backends)

# TODO [HSC]: remove when release
# train
# cd dectection/
# export CUDA_VISIBLE_DEVICES=5,4,3,2,1,0
# bash ./tools/dist_train.sh configs/litevla/mask_rcnn_litevla_m.py 4 --work-dir work_dirs/litevla_m

