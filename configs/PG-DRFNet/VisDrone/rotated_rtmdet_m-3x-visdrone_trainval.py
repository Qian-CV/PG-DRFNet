_base_ = ['../../../rotated_rtmdet/_base_/default_runtime.py', '../../../rotated_rtmdet/_base_/schedule_3x.py',
          '../../../_base_/datasets/visdrone_coco_trainval.py']

checkpoint = '/media/ubuntu/nvidia/wlq/part1_tiny_detection/mmrotate-1.x/tools/data/weight/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth' # noqa
base_lr = 0.004 / 32
angle_version = 'le90'
interval = 3

model = dict(
    type='PG_DRFNet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        out_indices=(1, 2, 3, 4),
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='DRF_FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=192,
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='CombinationSepBNHead',
        num_classes=12,
        in_channels=192,
        stacked_convs=2,
        feat_channels=192,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='mmdet.DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')

# batch_size = (1 GPUs) x (8 samples per GPU) = 8
train_dataloader = dict(batch_size=7, num_workers=7)
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = dict(batch_size=8, num_workers=8)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    # accumulative_counts=2,
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                    interval=interval,
                    max_keep_ckpts=6,
                    save_best='auto',
                    rule='greater'))
train_cfg = dict(
    type='EpochBasedTrainLoop', val_interval=interval)
work_dir = './work_dirs/visdrone/rotated_rtmdet_m-3x-visdrone_trainval/'
