_base_ = ['../../../../rotated_rtmdet/_base_/default_runtime.py', '../../../../rotated_rtmdet/_base_/schedule_6x.py',
          '../../../../_base_/datasets/vedai.py']

base_lr = 0.004 / 16  # 最初是16
interval = 4  # 最初是12
angle_version = 'le90'
checkpoint = '/media/ubuntu/nvidia/wlq/part1_tiny_detection/mmrotate-1.x/tools/data/weight/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth' # noqa
# fp16 = dict(loss_scale='dynamic')
model = dict(
    type='RTMDetGuidance',
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
        type='mmdet.CSPNeXtPAFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=192,
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='RotatedRTMDetGuidanceSepBNHead',
        num_classes=6,
        in_channels=192,
        stacked_convs=2,
        feat_channels=192,
        angle_version=angle_version,
        query_head_size=[192, 192, 4, 1],  # 对应修改前两位置的通道数为输入通道数in_channels
        query_layer_train=[1],  # 添加了索引层，找小目标的层
        small_object_scale=[[0, 32]],  # 添加了小目标范围
        small_center_dis_coeff=[1.],  # 添加了小目标中心
        query_loss_gammas=[1.2],  # 添加了损失函数超参数
        query_loss_weights=[10.],  # 添加了损失函数超参数
        query_infer=False,  # 是否使用快速推理方法
        query_threshold=0.12,
        infer_version='v3',
        context=4,
        layers_whole_test=[1, 2, 3],  # 不参与小目标索引的第分辨率特征层p4-p7一共四层
        layers_key_test=[1],  # 特征层的p3,p4作为钥匙来查找p2,p3
        layers_value_test=[0],  # 特征层的p3,p4作为钥匙来查找p2,p3
        loss_query_weight=1,  # 为了平横损失差距，对query部分损失缩放
        cls_layer_weight=[1.0, 1.0, 1.0, 1.0],  # 分类损失不同层权重占比，防止小目标主导剃度下降
        # cls_layer_weight=[1.0, 2.6, 1.2, 1.0],  # 分类损失不同层权重占比，防止小目标主导剃度下降
        reg_layer_weight=[1.0, 1.0, 1.0, 1.0],  # 回归损失不同层权重占比，防止小目标主导剃度下降
        # reg_layer_weight=[1.0, 2.6, 1.2, 1.0],  # 回归损失不同层权重占比，防止小目标主导剃度下降
        # cls_layer_weight=[2.5, 2.1, 1.4, 1],  # 分类损失不同层权重占比，防止小目标主导剃度下降
        # reg_layer_weight=[2.5, 2.1, 1.4, 1],  # 回归损失不同层权重占比，防止小目标主导剃度下降
        # cls_layer_weight='dynamic',  # 分类损失不同层权重占比，防止小目标主导剃度下降
        # reg_layer_weight='dynamic',  # 回归损失不同层权重占比，防止小目标主导剃度下降
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
            type='mmdet.DynamicSoftLabelAssigner',
            iou_calculator=dict(type='RBboxOverlaps2D'),
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

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr/2, weight_decay=0.05),
    # accumulative_counts=2,
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=interval, max_keep_ckpts=6))
train_cfg = dict(
    type='EpochBasedTrainLoop', val_interval=4)
work_dir = './work_dirs/ablation_study/A_study3/test2_1PGH/rotated_rtmdet_m_ch192-6x-vedai-4layers-guidancehead*1-32size/'
# work_dir = './work_dirs/shishi/'