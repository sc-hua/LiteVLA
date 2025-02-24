_base_ = ['./fpn_litevla_4xb4-80k_ade20k-512x512.py']

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_LITEVLA',
        version='litevla_n',
        backbone=True,
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/litevla_n.pth"  # TODO [HSC]: change to github release url later
    ),
    neck=dict(in_channels=(48, 96, 192, 384)),  # channels of each stage
    decode_head=dict(num_classes=150),
)

# TODO [HSC]: remove when release
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='seg_sem-fpn',
            name='litevla_n',
        )
    )]
visualizer = dict(vis_backends=vis_backends)

# TODO [HSC]: remove when release
# train
# cd segmentation/
# export CUDA_VISIBLE_DEVICES=5,4,3,2,1,0
# PORT=29509 bash ./tools/dist_train.sh configs/litevla/fpn_litevla_n.py 4 --work-dir output/litevla_n
