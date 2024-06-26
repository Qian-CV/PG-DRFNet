_base_ = ['../../../rotated_rtmdet/_base_/default_runtime.py', '../../../rotated_rtmdet/_base_/schedule_6x.py',
          '../../../_base_/datasets/dotav2_rr.py']

angle_version = 'le90'
checkpoint = '/media/ubuntu/nvidia/wlq/part1_tiny_detection/mmrotate-1.x/tools/data/weight/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth' # noqa
# fp16 = dict(loss_scale='dynamic')
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
        num_classes=18,
        in_channels=192,
        stacked_convs=2,
        feat_channels=192,
        angle_version=angle_version,
        guidance_head_size=[192, 192, 4, 1],
        guidance_layer_train=[1, 2],
        fp_object_scale=[[0, 32], [0, 64]],
        fp_center_dis_coeff=[1., 1.],
        guidance_loss_gammas=[1.2, 1.2],
        guidance_loss_weights=[10., 10.],
        dynamic_perception_infer=False,
        dp_version='v2',
        dp_threshold=0.12,
        context=4,
        layers_no_dp=[1, 2, 3],
        layers_key_test=[1, 2],
        layers_value_test=[0, 1],
        loss_guidance_weight=1,
        # cls_layer_weight=[1.0, 1.0, 1.0, 1.0],
        # reg_layer_weight=[1.0, 1.0, 1.0, 1.0],
        cls_layer_weight=[2.5, 2.1, 1.4, 1],
        reg_layer_weight=[2.5, 2.1, 1.4, 1],
        # cls_layer_weight='dynamic',
        # reg_layer_weight='dynamic',
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        # reg_decoded_bbox=True,
        # loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssignerSaveMemory',
            iou_calculator=dict(type='RBboxOverlaps2D'),
            gpu_assign_thr=1000,
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000),
)


# batch_size = (1 GPUs) x (8 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=8)
test_dataloader = dict(batch_size=8, num_workers=8)

base_lr = 0.004 / 32
interval = 4
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    accumulative_counts=2,
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=interval, max_keep_ckpts=6))
train_cfg = dict(
    type='EpochBasedTrainLoop', val_interval=4)
work_dir = './work_dirs/DOTA2_0/rtmdet/pg_drfnet-6x-dota2/'