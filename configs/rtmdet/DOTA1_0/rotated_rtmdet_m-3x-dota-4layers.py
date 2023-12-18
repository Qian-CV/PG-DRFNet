_base_ = '../rotated_rtmdet/rotated_rtmdet_l-6x-dota.py'
base_lr = 0.004 / 32  # 最初是16
interval = 8  # 最初是12

checkpoint = '/media/ubuntu/nvidia/wlq/part1_tiny_detection/mmrotate-1.x/tools/data/weight/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth' # noqa
# fp16 = dict(loss_scale='dynamic')
model = dict(
    backbone=dict(
        out_indices=(1, 2, 3, 4),  # add
        deepen_factor=0.67,
        widen_factor=0.75,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384, 768], out_channels=96, num_csp_blocks=2),
    bbox_head=dict(
        in_channels=96,
        feat_channels=96,
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[4, 8, 16, 32])))

# batch_size = (1 GPUs) x (8 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=2)
val_dataloader = dict(batch_size=2, num_workers=8)
test_dataloader = dict(batch_size=8, num_workers=8)

# optimizer
optim_wrapper = dict(
    # type='OptimWrapper',
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    accumulative_counts=2,
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=interval, max_keep_ckpts=9))
work_dir = './work_dirs/rotated_rtmdet_m_4layers-6x-dota-gradient_accumulation/'
# work_dir = './work_dirs/shishi/'